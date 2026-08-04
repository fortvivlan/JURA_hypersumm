"""QLoRA training, validation, and full RAG document-inference workflow."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .colab_support import (
    deliver_file,
    get_huggingface_token,
    prepare_artifact_root,
    selected_docx_files,
)
from .common import (
    DEFAULT_DRIVE_ROOT,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_RAG_DIR,
    DEFAULT_RAG_REVISION,
    DEFAULT_RESULTS_DIR,
    Task,
    announce_stage,
    configure_reproducibility,
    default_dataset_path,
    evaluate_predictions,
    file_sha256,
    load_saved_artifact_manifest,
    load_dataset,
    merge_parameters,
    prompt_sha256,
    reproducibility_metadata,
    resolve_huggingface_revision,
    resolve_model,
    saved_artifact_revision,
    slugify_model_id,
    validate_task,
)
from .inference import run_document_inference
from .llm_common import CausalPredictor, load_causal_model
from .prompting import (
    build_ministral_training_texts,
    build_training_texts,
    prompt_for_task,
)
from .reporting import (
    display_scores,
    write_document_review_package,
    write_results_workbook,
)
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
    "embedding_revision": DEFAULT_EMBEDDING_REVISION,
    "seed": 42,
    "deterministic": True,
}

STANDARD_PROMPT_PROCESSING = "standard_chat_template_v1"
MINISTRAL_PROMPT_PROCESSING = "ministral_duplicated_system_prompt_v1"


def _prompt_processing_strategy(model_alias: str) -> str:
    return (
        MINISTRAL_PROMPT_PROCESSING
        if model_alias == "ministral"
        else STANDARD_PROMPT_PROCESSING
    )


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        length += 1
    return length


def _tokenize_training_rows(
    dataframe,
    tokenizer,
    task: Task,
    max_length: int,
    *,
    model_alias: str,
):
    """Tokenize SFT rows while preserving both prompt context and label tokens."""
    from tqdm.auto import tqdm

    prompt_processing = _prompt_processing_strategy(model_alias)
    build_texts = (
        build_ministral_training_texts
        if prompt_processing == MINISTRAL_PROMPT_PROCESSING
        else build_training_texts
    )
    rows: list[dict[str, list[int]]] = []
    for row in tqdm(
        dataframe.itertuples(index=False),
        total=len(dataframe),
        desc=f"Formatting {task} training data",
    ):
        prompt, full = build_texts(
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
        model_alias=spec.alias,
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
        full_determinism=bool(parameters["deterministic"]),
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


def _load_saved_adapter(
    adapter_dir: Path,
    *,
    spec,
    revision: str,
    token: str | None,
    parameters: Mapping[str, Any],
):
    """Load a saved LoRA adapter over its immutable base-model revision."""
    from peft import PeftModel
    from transformers import AutoTokenizer

    loaded = load_causal_model(
        spec,
        revision=revision,
        token=token,
        quantization=bool(parameters["quantization"]),
        device_map=parameters["device_map"],
        precision=str(parameters["precision"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        local_files_only=True,
        trust_remote_code=spec.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = PeftModel.from_pretrained(
        loaded.model,
        adapter_dir,
        is_trainable=False,
        local_files_only=True,
    )
    model.config.use_cache = True
    model.eval()
    return model, tokenizer, loaded.compute_dtype


def run(
    model_name: str,
    task: str,
    hyperparameters: Mapping[str, Any] | None = None,
    *,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
    rag_dir: str | Path = DEFAULT_RAG_DIR,
    rag_revision: str = DEFAULT_RAG_REVISION,
    drive_root: str | Path = DEFAULT_DRIVE_ROOT,
    revision: str | None = None,
    use_existing_model: bool = False,
    document_paths: Sequence[str | Path] | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
):
    """Train or reuse, validate, and document-test one task LoRA adapter.

    ``hyperparameters`` may override any key in
    :data:`DEFAULT_LORA_HYPERPARAMETERS`; unknown keys are rejected.
    With ``use_existing_model=True``, the compatible final adapter in artifact
    storage is required and training is skipped. Outside Colab, pass explicit
    local ``drive_root``, ``rag_dir``, ``results_dir``, and ``document_paths``.
    The function returns the main validation score table and downloads a
    detailed XLSX workbook.
    """
    validated_task = validate_task(task)
    spec = resolve_model(model_name)
    prompt_processing = _prompt_processing_strategy(spec.alias)
    workflow = f"lora/{spec.alias}/{validated_task}"
    announce_stage(workflow, "setup", "Preparing datasets and artifact paths.")
    train_path = Path(train_path or default_dataset_path("train", validated_task))
    val_path = Path(val_path or default_dataset_path("val", validated_task))
    train_hash = file_sha256(train_path)
    val_hash = file_sha256(val_path)
    current_prompt_hash = prompt_sha256(prompt_for_task(validated_task))
    project_drive = prepare_artifact_root(drive_root)
    adapter_target = (
        project_drive
        / "models"
        / "lora"
        / slugify_model_id(spec.model_id)
        / validated_task
    )
    artifact_manifest: dict[str, Any] | None = None
    if use_existing_model:
        announce_stage(
            workflow,
            "reuse",
            f"Checking the previously trained adapter at {adapter_target}.",
        )
        artifact_manifest = load_saved_artifact_manifest(
            adapter_target,
            required_files=(
                "run_config.json",
                "adapter_config.json",
                "tokenizer_config.json",
            ),
            weight_files=("adapter_model.safetensors", "adapter_model.bin"),
            expected={
                "model_id": spec.model_id,
                "task": validated_task,
                "train_sha256": train_hash,
                "validation_sha256": val_hash,
                "prompt_sha256": current_prompt_hash,
                "prompt_processing": prompt_processing,
            },
        )
        stored_parameters = artifact_manifest.get("hyperparameters")
        if not isinstance(stored_parameters, dict):
            raise ValueError(
                f"Saved LoRA manifest has no hyperparameters: {adapter_target}"
            )
        parameters = merge_parameters(DEFAULT_LORA_HYPERPARAMETERS, stored_parameters)
        parameters = merge_parameters(parameters, hyperparameters)
    else:
        parameters = merge_parameters(DEFAULT_LORA_HYPERPARAMETERS, hyperparameters)
    configure_reproducibility(
        int(parameters["seed"]),
        deterministic=bool(parameters["deterministic"]),
    )
    reproducibility = reproducibility_metadata(
        int(parameters["seed"]),
        deterministic=bool(parameters["deterministic"]),
    )
    val_dataframe = load_dataset(val_path, validated_task)
    token = get_huggingface_token()
    if artifact_manifest is not None:
        effective_revision = saved_artifact_revision(
            artifact_manifest, adapter_target
        )
        if revision is not None:
            requested_commit = resolve_huggingface_revision(
                spec.model_id, revision, token=token
            )
            if requested_commit != effective_revision:
                raise ValueError(
                    "Requested base-model revision does not match the saved adapter: "
                    f"{requested_commit} != {effective_revision}"
                )
    else:
        effective_revision = resolve_huggingface_revision(
            spec.model_id, revision or spec.revision, token=token
        )
    announce_stage(workflow, "rag", "Preparing the pinned RAG repository.")
    rag_path, rag_commit = ensure_rag_repository(
        rag_dir, revision=rag_revision
    )
    announce_stage(workflow, "rag", f"RAG repository ready at {rag_commit[:12]}.")
    model = tokenizer = None
    try:
        if artifact_manifest is not None:
            announce_stage(
                workflow,
                "load",
                "Loading the saved LoRA adapter; training is skipped.",
            )
            model, tokenizer, compute_dtype = _load_saved_adapter(
                adapter_target,
                spec=spec,
                revision=effective_revision,
                token=token,
                parameters=parameters,
            )
            history = artifact_manifest.get("training_history", [])
            training_metrics = artifact_manifest.get("training_metrics", {})
            announce_stage(workflow, "load", "Saved LoRA adapter loaded.")
        else:
            train_dataframe = load_dataset(train_path, validated_task)
            announce_stage(
                workflow,
                "training",
                f"Starting LoRA fine-tuning on {len(train_dataframe)} examples.",
            )
            model, tokenizer, compute_dtype, history, training_metrics = _train_adapter(
                train_dataframe,
                spec=spec,
                task=validated_task,
                revision=effective_revision,
                token=token,
                parameters=parameters,
            )
            adapter_target.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_target, safe_serialization=True)
            tokenizer.save_pretrained(adapter_target)
            (adapter_target / "run_config.json").write_text(
                json.dumps(
                    {
                        "model_id": spec.model_id,
                        "requested_revision": revision or "registry_default",
                        "resolved_revision": effective_revision,
                        "task": validated_task,
                        "hyperparameters": parameters,
                        "training_history": history,
                        "training_metrics": training_metrics,
                        "train_sha256": train_hash,
                        "validation_sha256": val_hash,
                        "prompt_sha256": current_prompt_hash,
                        "prompt_processing": prompt_processing,
                        "rag_requested_revision": rag_revision,
                        "rag_revision": rag_commit,
                        "reproducibility": reproducibility,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            announce_stage(
                workflow,
                "training",
                f"Fine-tuning finished; final adapter saved to {adapter_target}.",
            )
        announce_stage(
            workflow,
            "validation",
            f"Starting validation on {len(val_dataframe)} examples.",
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
            progress_description=f"Validating LoRA {validated_task}",
        )
        evaluation = evaluate_predictions(
            val_dataframe,
            [item.label for item in generated],
            [item.raw_output for item in generated],
            model_id=spec.model_id,
            task=validated_task,
        )
        announce_stage(workflow, "validation", "Validation finished.")
        announce_stage(workflow, "testing", "Loading the RAG retrieval index.")
        retriever = PremiseRetriever.from_rag_directory(
            rag_path,
            embedding_revision=str(parameters["embedding_revision"]),
            embedding_device=str(parameters["embedding_device"]),
        )
        announce_stage(workflow, "testing", "RAG retrieval index loaded.")
        document_predictor = CausalPredictor(
            model,
            tokenizer,
            validated_task,
            batch_size=int(parameters["document_batch_size"]),
            max_input_length=int(parameters["max_input_length"]),
            max_new_tokens=int(parameters["max_new_tokens"]),
        )
        announce_stage(
            workflow,
            "testing",
            "Document testing is ready. Select or provide DOCX files, or skip it.",
        )
        with selected_docx_files(document_paths) as documents:
            if documents:
                announce_stage(
                    workflow,
                    "testing",
                    f"Starting full RAG inference for {len(documents)} document(s).",
                )
            else:
                announce_stage(workflow, "testing", "No DOCX files selected; testing skipped.")
            document_tables = run_document_inference(
                documents,
                predictor=document_predictor,
                retriever=retriever,
                model_id=spec.model_id,
                task=validated_task,
                top_k=int(parameters["retrieval_top_k"]),
            )
        if documents:
            announce_stage(workflow, "testing", "Full document inference finished.")
        import pandas as pd

        announce_stage(workflow, "results", "Writing score and review artifacts.")
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
                "requested_revision": revision or (
                    "saved_artifact" if artifact_manifest is not None else "registry_default"
                ),
                "resolved_revision": effective_revision,
                "task": validated_task,
                "parameters": parameters,
                "training_metrics": training_metrics,
                "train_sha256": train_hash,
                "validation_sha256": val_hash,
                "prompt_sha256": current_prompt_hash,
                "prompt_processing": prompt_processing,
                "rag_requested_revision": rag_revision,
                "rag_commit": rag_commit,
                "compute_dtype": compute_dtype,
                "summary_enabled": False,
                "drive_adapter_path": adapter_target,
                "artifact_path": adapter_target,
                "used_existing_model": artifact_manifest is not None,
                "training_skipped": artifact_manifest is not None,
                "artifact_hyperparameters": (
                    artifact_manifest.get("hyperparameters")
                    if artifact_manifest is not None
                    else None
                ),
                "reproducibility": reproducibility,
                "bitwise_reproducibility_scope": "same code, lockfile, GPU model, driver, and CUDA stack",
            },
            output_dir=results_dir,
        )
        review_package = write_document_review_package(
            f"lora_{slugify_model_id(spec.model_id)}_{validated_task}",
            document_tables.pairs,
            output_dir=results_dir,
        )
        display_scores(evaluation.scores)
        deliver_file(workbook)
        if review_package is not None:
            deliver_file(review_package)
        announce_stage(workflow, "complete", "Workflow finished.")
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
