"""QLoRA training, validation, and full RAG document-inference workflow."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Mapping

from .colab_support import (
    download_file,
    get_huggingface_token,
    mount_drive,
    require_colab,
    uploaded_docx_files,
)
from .common import (
    DEFAULT_DRIVE_ROOT,
    DEFAULT_RAG_DIR,
    DEFAULT_RESULTS_DIR,
    Task,
    default_dataset_path,
    evaluate_predictions,
    file_sha256,
    load_dataset,
    merge_parameters,
    prompt_sha256,
    resolve_model,
    set_random_seed,
    slugify_model_id,
    validate_task,
)
from .inference import run_document_inference
from .llm_common import CausalPredictor, load_causal_model
from .prompting import build_training_texts, prompt_for_task
from .reporting import display_scores, write_results_workbook
from .retrieval import PremiseRetriever, ensure_rag_repository

DEFAULT_LORA_HYPERPARAMETERS: dict[str, Any] = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",
    "max_seq_length": 1024,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "epochs": 3,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True,
    "logging_steps": 20,
    "num_workers": 0,
    "quantization": True,
    "device_map": "auto",
    "precision": "auto",
    "inference_batch_size": 1,
    "document_batch_size": 8,
    "max_input_length": 4096,
    "max_new_tokens": 16,
    "retrieval_top_k": 20,
    "embedding_device": "cpu",
    "seed": 42,
}


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        length += 1
    return length


def _tokenize_training_rows(dataframe, tokenizer, task: Task, max_length: int):
    """Tokenize SFT rows while preserving both prompt context and label tokens."""
    from tqdm.auto import tqdm

    rows: list[dict[str, list[int]]] = []
    for row in tqdm(
        dataframe.itertuples(index=False),
        total=len(dataframe),
        desc=f"Formatting {task} training data",
    ):
        prompt, full = build_training_texts(
            tokenizer, row.premise, row.hypothesis, row.tag, task
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        common_length = _common_prefix_length(prompt_ids, full_ids)
        response_ids = full_ids[common_length:]
        if not response_ids:
            response_ids = tokenizer(
                f" {row.tag}{tokenizer.eos_token or ''}", add_special_tokens=False
            )["input_ids"]
        response_ids = response_ids[:max_length]
        prompt_budget = max_length - len(response_ids)
        if len(prompt_ids) > prompt_budget:
            prefix_length = prompt_budget // 2
            suffix_length = prompt_budget - prefix_length
            prompt_ids = (
                prompt_ids[:prefix_length]
                + (prompt_ids[-suffix_length:] if suffix_length else [])
            )
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + list(response_ids)
        if not any(label != -100 for label in labels):
            raise ValueError("Training example has no assistant-label tokens")
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    return rows


def _train_adapter(
    dataframe,
    *,
    spec,
    task: Task,
    revision: str,
    token: str | None,
    parameters: Mapping[str, Any],
):
    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from torch.utils.data import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    loaded = load_causal_model(
        spec,
        revision=revision,
        token=token,
        quantization=bool(parameters["quantization"]),
        device_map=parameters["device_map"],
        precision=str(parameters["precision"]),
    )
    tokenizer = loaded.tokenizer
    tokenizer.padding_side = "right"
    model = loaded.model
    model.config.use_cache = False
    if bool(parameters["quantization"]):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(parameters["gradient_checkpointing"]),
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(parameters["lora_rank"]),
            lora_alpha=int(parameters["lora_alpha"]),
            target_modules=parameters["target_modules"],
            lora_dropout=float(parameters["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    tokenized_rows = _tokenize_training_rows(
        dataframe,
        tokenizer,
        task,
        int(parameters["max_seq_length"]),
    )

    class TokenDataset(Dataset):
        def __len__(self):
            return len(tokenized_rows)

        def __getitem__(self, index):
            return tokenized_rows[index]

    use_bf16 = loaded.compute_dtype == "bfloat16"
    training_args = TrainingArguments(
        output_dir="/content/jura_lora_trainer" if Path("/content").is_dir() else "./jura_lora_trainer",
        num_train_epochs=float(parameters["epochs"]),
        per_device_train_batch_size=int(parameters["batch_size"]),
        gradient_accumulation_steps=int(parameters["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(parameters["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if bool(parameters["quantization"]) else "adamw_torch",
        learning_rate=float(parameters["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=float(parameters["warmup_ratio"]),
        weight_decay=float(parameters["weight_decay"]),
        max_grad_norm=float(parameters["max_grad_norm"]),
        logging_steps=int(parameters["logging_steps"]),
        save_strategy="no",
        eval_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=int(parameters["num_workers"]),
        seed=int(parameters["seed"]),
        data_seed=int(parameters["seed"]),
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenDataset(),
        data_collator=collator,
    )
    train_output = trainer.train()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    tokenizer.padding_side = "left"
    history = list(trainer.state.log_history)
    training_metrics = dict(train_output.metrics)
    del trainer, train_output
    gc.collect()
    torch.cuda.empty_cache()
    return model, tokenizer, loaded.compute_dtype, history, training_metrics


def run(
    model_name: str,
    task: str,
    hyperparameters: Mapping[str, Any] | None = None,
    *,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
    rag_dir: str | Path = DEFAULT_RAG_DIR,
    drive_root: str | Path = DEFAULT_DRIVE_ROOT,
    revision: str = "main",
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
):
    """Train, validate, save, and document-test one supported task LoRA adapter.

    ``hyperparameters`` may override any key in
    :data:`DEFAULT_LORA_HYPERPARAMETERS`; unknown keys are rejected.
    The function returns the main validation score table and downloads a
    detailed XLSX workbook in Google Colab.
    """
    require_colab()
    validated_task = validate_task(task)
    spec = resolve_model(model_name)
    parameters = merge_parameters(DEFAULT_LORA_HYPERPARAMETERS, hyperparameters)
    set_random_seed(int(parameters["seed"]))
    train_path = Path(train_path or default_dataset_path("train", validated_task))
    val_path = Path(val_path or default_dataset_path("val", validated_task))
    train_dataframe = load_dataset(train_path, validated_task)
    val_dataframe = load_dataset(val_path, validated_task)
    rag_path, rag_commit = ensure_rag_repository(rag_dir)
    token = get_huggingface_token()
    model = tokenizer = None
    try:
        model, tokenizer, compute_dtype, history, training_metrics = _train_adapter(
            train_dataframe,
            spec=spec,
            task=validated_task,
            revision=revision,
            token=token,
            parameters=parameters,
        )
        project_drive = mount_drive(drive_root)
        adapter_target = (
            project_drive
            / "models"
            / "lora"
            / slugify_model_id(spec.model_id)
            / validated_task
        )
        adapter_target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_target, safe_serialization=True)
        tokenizer.save_pretrained(adapter_target)
        (adapter_target / "run_config.json").write_text(
            json.dumps(
                {
                    "model_id": spec.model_id,
                    "revision": revision,
                    "task": validated_task,
                    "hyperparameters": parameters,
                    "training_metrics": training_metrics,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        predictor = CausalPredictor(
            model,
            tokenizer,
            validated_task,
            batch_size=int(parameters["inference_batch_size"]),
            max_input_length=int(parameters["max_input_length"]),
            max_new_tokens=int(parameters["max_new_tokens"]),
        )
        generated = predictor.predict_examples(
            val_dataframe["premise"].tolist(),
            val_dataframe["hypothesis"].tolist(),
        )
        evaluation = evaluate_predictions(
            val_dataframe,
            [item.label for item in generated],
            [item.raw_output for item in generated],
            model_id=spec.model_id,
            task=validated_task,
        )
        retriever = PremiseRetriever.from_rag_directory(
            rag_path, embedding_device=str(parameters["embedding_device"])
        )
        document_predictor = CausalPredictor(
            model,
            tokenizer,
            validated_task,
            batch_size=int(parameters["document_batch_size"]),
            max_input_length=int(parameters["max_input_length"]),
            max_new_tokens=int(parameters["max_new_tokens"]),
        )
        print("Upload one or more .docx court decisions (Cancel to skip document testing).")
        with uploaded_docx_files() as documents:
            document_tables = run_document_inference(
                documents,
                predictor=document_predictor,
                retriever=retriever,
                model_id=spec.model_id,
                task=validated_task,
                top_k=int(parameters["retrieval_top_k"]),
            )
        import pandas as pd

        workbook = write_results_workbook(
            f"lora_{slugify_model_id(spec.model_id)}_{validated_task}",
            {
                "scores": evaluation.scores,
                "per_class": evaluation.per_class,
                "confusion_matrix": evaluation.confusion_matrix,
                "validation_predictions": evaluation.predictions,
                "training_history": pd.DataFrame(history),
                "document_aggregates": document_tables.aggregates,
                "document_pairs": document_tables.pairs,
                "errors": document_tables.errors,
            },
            {
                "workflow": "lora",
                "model_id": spec.model_id,
                "requested_revision": revision,
                "resolved_revision": getattr(model.config, "_commit_hash", None),
                "task": validated_task,
                "parameters": parameters,
                "training_metrics": training_metrics,
                "train_sha256": file_sha256(train_path),
                "validation_sha256": file_sha256(val_path),
                "prompt_sha256": prompt_sha256(prompt_for_task(validated_task)),
                "rag_commit": rag_commit,
                "compute_dtype": compute_dtype,
                "summary_enabled": False,
                "drive_adapter_path": adapter_target,
                "remaining_nondeterminism": "CUDA kernels and quantized training may vary by hardware/library version",
            },
            output_dir=results_dir,
        )
        display_scores(evaluation.scores)
        download_file(workbook)
        return evaluation.scores
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass
