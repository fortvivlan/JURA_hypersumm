"""BERT baseline training, validation, and full RAG document inference."""

from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .colab_support import download_file, mount_drive, require_colab, uploaded_docx_files
from .common import (
    DEFAULT_DRIVE_ROOT,
    DEFAULT_RAG_DIR,
    DEFAULT_RESULTS_DIR,
    LABEL2ID_BY_TASK,
    Task,
    default_dataset_path,
    evaluate_predictions,
    file_sha256,
    load_dataset,
    merge_parameters,
    set_random_seed,
)
from .inference import ModelPrediction, run_document_inference
from .reporting import (
    display_scores,
    write_document_review_package,
    write_results_workbook,
)
from .retrieval import PremiseRetriever, ensure_rag_repository

DEFAULT_BERT_MODEL = "ai-forever/sbert_large_nlu_ru"
DEFAULT_BERT_PARAMETERS: dict[str, Any] = {
    "max_length": 512,
    "batch_size": 16,
    "inference_batch_size": 32,
    "epochs": 6,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "num_workers": 2,
    "mixed_precision": True,
    "precision": "auto",
    "device": "auto",
    "gradient_checkpointing": False,
    "retrieval_top_k": 20,
    "embedding_device": "cpu",
    "seed": 42,
}


class BertPredictor:
    """Batched sequence-classifier predictor."""

    def __init__(self, model, tokenizer, *, batch_size: int, max_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def predict_examples(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> list[ModelPrediction]:
        if len(premises) != len(hypotheses):
            raise ValueError("Premises and hypotheses must have equal length")
        import torch

        predictions: list[ModelPrediction] = []
        for start in range(0, len(premises), self.batch_size):
            encoded = self.tokenizer(
                list(premises[start : start + self.batch_size]),
                list(hypotheses[start : start + self.batch_size]),
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.no_grad():
                ids = self.model(**encoded).logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(
                ModelPrediction(self.model.config.id2label[int(label_id)])
                for label_id in ids
            )
        return predictions

    def predict_pairs(
        self, premises: Sequence[str], hypothesis: str
    ) -> list[ModelPrediction]:
        return self.predict_examples(premises, [hypothesis] * len(premises))


def _train_bert(
    train_dataframe,
    val_dataframe,
    *,
    task: Task,
    model_id: str,
    revision: str,
    parameters: Mapping[str, Any],
):
    import torch
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    requested_device = str(parameters["device"]).lower()
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but is not available")
    label2id = LABEL2ID_BY_TASK[task]
    id2label = {index: label for label, index in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    ).to(device)
    if bool(parameters["gradient_checkpointing"]):
        model.gradient_checkpointing_enable()

    class PairDataset(Dataset):
        def __init__(self, dataframe):
            self.rows = dataframe.reset_index(drop=True)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows.iloc[index]
            return row["premise"], row["hypothesis"], label2id[row["tag"]]

    def collate(rows):
        premises, hypotheses, labels = zip(*rows)
        encoded = tokenizer(
            list(premises),
            list(hypotheses),
            max_length=int(parameters["max_length"]),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    generator = torch.Generator().manual_seed(int(parameters["seed"]))
    train_loader = DataLoader(
        PairDataset(train_dataframe),
        batch_size=int(parameters["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=int(parameters["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        PairDataset(val_dataframe),
        batch_size=int(parameters["inference_batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=int(parameters["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    total_steps = len(train_loader) * int(parameters["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(parameters["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    use_precision = bool(parameters["mixed_precision"]) and device.type == "cuda"
    requested_precision = str(parameters["precision"]).lower()
    if requested_precision not in {"auto", "bfloat16", "float16"}:
        raise ValueError("precision must be 'auto', 'bfloat16', or 'float16'")
    use_bf16 = requested_precision == "bfloat16" or (
        requested_precision == "auto"
        and device.type == "cuda"
        and bool(torch.cuda.is_bf16_supported())
    )
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bfloat16 was requested but is unsupported by this GPU")
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_precision and not use_bf16
    )
    history: list[dict[str, Any]] = []
    temp_parent = "/content" if Path("/content").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="jura_bert_best_", dir=temp_parent) as temp:
        best_path = Path(temp) / "best_state.pt"
        best_f1 = -1.0
        for epoch in range(1, int(parameters["epochs"]) + 1):
            model.train()
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"BERT {task} epoch {epoch}"):
                batch = {
                    key: value.to(device, non_blocking=device.type == "cuda")
                    for key, value in batch.items()
                }
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype, enabled=use_precision
                ):
                    output = model(**batch)
                    loss = output.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(parameters["max_grad_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += float(loss.item())

            model.eval()
            gold_ids: list[int] = []
            predicted_ids: list[int] = []
            with torch.no_grad():
                for batch in val_loader:
                    gold_ids.extend(batch["labels"].tolist())
                    gpu_batch = {
                        key: value.to(device, non_blocking=device.type == "cuda")
                        for key, value in batch.items()
                        if key != "labels"
                    }
                    predicted_ids.extend(
                        model(**gpu_batch).logits.argmax(dim=-1).cpu().tolist()
                    )
            macro_f1 = f1_score(gold_ids, predicted_ids, average="macro")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(1, len(train_loader)),
                    "validation_macro_f1": macro_f1,
                }
            )
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), best_path)
        try:
            state = torch.load(best_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(best_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    compute_dtype = "float32" if not use_precision else ("bfloat16" if use_bf16 else "float16")
    return model, tokenizer, history, compute_dtype


def _run_bert(
    task: Task,
    *,
    train_path: str | Path | None,
    val_path: str | Path | None,
    rag_dir: str | Path,
    drive_root: str | Path,
    model_id: str,
    revision: str,
    hyperparameters: Mapping[str, Any] | None,
    results_dir: str | Path,
):
    require_colab()
    parameters = merge_parameters(DEFAULT_BERT_PARAMETERS, hyperparameters)
    set_random_seed(int(parameters["seed"]))
    train_path = Path(train_path or default_dataset_path("train", task))
    val_path = Path(val_path or default_dataset_path("val", task))
    train_dataframe = load_dataset(train_path, task)
    val_dataframe = load_dataset(val_path, task)
    rag_path, rag_commit = ensure_rag_repository(rag_dir)
    model = tokenizer = None
    try:
        model, tokenizer, history, compute_dtype = _train_bert(
            train_dataframe,
            val_dataframe,
            task=task,
            model_id=model_id,
            revision=revision,
            parameters=parameters,
        )
        project_drive = mount_drive(drive_root)
        model_target = project_drive / "models" / "bert" / task
        model_target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_target, safe_serialization=True)
        tokenizer.save_pretrained(model_target)
        (model_target / "run_config.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "task": task,
                    "hyperparameters": parameters,
                    "training_history": history,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        predictor = BertPredictor(
            model,
            tokenizer,
            batch_size=int(parameters["inference_batch_size"]),
            max_length=int(parameters["max_length"]),
        )
        validation = predictor.predict_examples(
            val_dataframe["premise"].tolist(),
            val_dataframe["hypothesis"].tolist(),
        )
        evaluation = evaluate_predictions(
            val_dataframe,
            [item.label for item in validation],
            [item.raw_output for item in validation],
            model_id=model_id,
            task=task,
        )
        retriever = PremiseRetriever.from_rag_directory(
            rag_path, embedding_device=str(parameters["embedding_device"])
        )
        print("Upload one or more .docx court decisions (Cancel to skip document testing).")
        with uploaded_docx_files() as documents:
            document_tables = run_document_inference(
                documents,
                predictor=predictor,
                retriever=retriever,
                model_id=model_id,
                task=task,
                top_k=int(parameters["retrieval_top_k"]),
            )
        workbook = write_results_workbook(
            f"bert_{task}",
            {
                "scores": evaluation.scores,
                "per_class": evaluation.per_class,
                "confusion_matrix": evaluation.confusion_matrix,
                "validation_predictions": evaluation.predictions,
                "training_history": __import__("pandas").DataFrame(history),
                "document_aggregates": document_tables.aggregates,
                "document_pairs": document_tables.pairs,
                "errors": document_tables.errors,
            },
            {
                "workflow": "bert",
                "model_id": model_id,
                "requested_revision": revision,
                "resolved_revision": getattr(model.config, "_commit_hash", None),
                "task": task,
                "parameters": parameters,
                "train_sha256": file_sha256(train_path),
                "validation_sha256": file_sha256(val_path),
                "rag_commit": rag_commit,
                "compute_dtype": compute_dtype,
                "summary_enabled": False,
                "drive_model_path": model_target,
                "remaining_nondeterminism": "CUDA kernels may vary by GPU and library version",
            },
            output_dir=results_dir,
        )
        review_package = write_document_review_package(
            f"bert_{task}",
            document_tables.pairs,
            output_dir=results_dir,
        )
        display_scores(evaluation.scores)
        download_file(workbook)
        if review_package is not None:
            download_file(review_package)
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


def run_bert_binary(
    *,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
    rag_dir: str | Path = DEFAULT_RAG_DIR,
    drive_root: str | Path = DEFAULT_DRIVE_ROOT,
    model_id: str = DEFAULT_BERT_MODEL,
    revision: str = "main",
    hyperparameters: Mapping[str, Any] | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
):
    """Train, validate, save, and document-test the binary BERT baseline."""
    return _run_bert(
        "binary",
        train_path=train_path,
        val_path=val_path,
        rag_dir=rag_dir,
        drive_root=drive_root,
        model_id=model_id,
        revision=revision,
        hyperparameters=hyperparameters,
        results_dir=results_dir,
    )


def run_bert_ternary(
    *,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
    rag_dir: str | Path = DEFAULT_RAG_DIR,
    drive_root: str | Path = DEFAULT_DRIVE_ROOT,
    model_id: str = DEFAULT_BERT_MODEL,
    revision: str = "main",
    hyperparameters: Mapping[str, Any] | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
):
    """Train, validate, save, and document-test the ternary BERT baseline."""
    return _run_bert(
        "ternary",
        train_path=train_path,
        val_path=val_path,
        rag_dir=rag_dir,
        drive_root=drive_root,
        model_id=model_id,
        revision=revision,
        hyperparameters=hyperparameters,
        results_dir=results_dir,
    )
