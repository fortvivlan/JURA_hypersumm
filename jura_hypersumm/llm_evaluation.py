"""Ready-LLM validation and full RAG document-inference workflow."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping

from .colab_support import (
    download_file,
    get_huggingface_token,
    require_colab,
    uploaded_docx_files,
)
from .common import (
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_RAG_DIR,
    DEFAULT_RAG_REVISION,
    DEFAULT_RESULTS_DIR,
    EvaluationTables,
    configure_reproducibility,
    default_dataset_path,
    evaluate_predictions,
    file_sha256,
    load_dataset,
    merge_parameters,
    prompt_sha256,
    reproducibility_metadata,
    resolve_huggingface_revision,
    resolve_model,
    slugify_model_id,
)
from .inference import run_document_inference
from .llm_common import CausalPredictor, load_causal_model
from .prompting import prompt_for_task
from .reporting import (
    concatenate_tables,
    display_scores,
    write_document_review_package,
    write_results_workbook,
)
from .retrieval import PremiseRetriever, ensure_rag_repository


DEFAULT_INFERENCE_PARAMETERS: dict[str, Any] = {
    "batch_size": 1,
    "document_batch_size": 8,
    "max_input_length": 4096,
    "max_new_tokens": 16,
    "quantization": True,
    "device_map": "auto",
    "precision": "auto",
    "retrieval_top_k": 20,
    "embedding_device": "cpu",
    "embedding_revision": DEFAULT_EMBEDDING_REVISION,
    "seed": 42,
    "deterministic": True,
}


def _validate_task(
    dataframe,
    *,
    model,
    tokenizer,
    model_id: str,
    task: str,
    parameters: Mapping[str, Any],
) -> tuple[EvaluationTables, CausalPredictor]:
    predictor = CausalPredictor(
        model,
        tokenizer,
        task,  # type: ignore[arg-type]
        batch_size=int(parameters["batch_size"]),
        max_input_length=int(parameters["max_input_length"]),
        max_new_tokens=int(parameters["max_new_tokens"]),
    )
    generated = predictor.predict_examples(
        dataframe["premise"].tolist(), dataframe["hypothesis"].tolist()
    )
    tables = evaluate_predictions(
        dataframe,
        [item.label for item in generated],
        [item.raw_output for item in generated],
        model_id=model_id,
        task=task,  # type: ignore[arg-type]
    )
    return tables, predictor


def run_llm_evaluation(
    model_name: str,
    *,
    val_binary_path: str | Path | None = None,
    val_ternary_path: str | Path | None = None,
    rag_dir: str | Path = DEFAULT_RAG_DIR,
    rag_revision: str = DEFAULT_RAG_REVISION,
    revision: str | None = None,
    inference_parameters: Mapping[str, Any] | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
):
    """Validate one supported ready LLM on both tasks and test uploaded DOCX files.

    The function is intended for Google Colab. It returns the main score table
    and downloads a detailed XLSX workbook containing validation and document
    inference audit tables.
    """
    require_colab()
    parameters = merge_parameters(DEFAULT_INFERENCE_PARAMETERS, inference_parameters)
    configure_reproducibility(
        int(parameters["seed"]),
        deterministic=bool(parameters["deterministic"]),
    )
    spec = resolve_model(model_name)
    binary_path = Path(val_binary_path or default_dataset_path("val", "binary"))
    ternary_path = Path(val_ternary_path or default_dataset_path("val", "ternary"))
    binary_hash = file_sha256(binary_path)
    ternary_hash = file_sha256(ternary_path)
    reproducibility = reproducibility_metadata(
        int(parameters["seed"]),
        deterministic=bool(parameters["deterministic"]),
    )
    datasets = {
        "binary": load_dataset(binary_path, "binary"),
        "ternary": load_dataset(ternary_path, "ternary"),
    }
    token = get_huggingface_token()
    effective_revision = resolve_huggingface_revision(
        spec.model_id, revision or spec.revision, token=token
    )
    rag_path, rag_commit = ensure_rag_repository(
        rag_dir, revision=rag_revision
    )
    loaded = load_causal_model(
        spec,
        revision=effective_revision,
        token=token,
        quantization=bool(parameters["quantization"]),
        device_map=parameters["device_map"],
        precision=str(parameters["precision"]),
    )
    validation_tables: list[EvaluationTables] = []
    predictors: dict[str, CausalPredictor] = {}
    try:
        for task in ("binary", "ternary"):
            tables, predictor = _validate_task(
                datasets[task],
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                model_id=spec.model_id,
                task=task,
                parameters=parameters,
            )
            validation_tables.append(tables)
            predictors[task] = predictor

        retriever = PremiseRetriever.from_rag_directory(
            rag_path,
            embedding_revision=str(parameters["embedding_revision"]),
            embedding_device=str(parameters["embedding_device"]),
        )
        document_aggregates = []
        document_pairs = []
        document_errors = []
        print("Upload one or more .docx court decisions (Cancel to skip document testing).")
        with uploaded_docx_files() as documents:
            for task in ("binary", "ternary"):
                document_predictor = CausalPredictor(
                    loaded.model,
                    loaded.tokenizer,
                    task,
                    batch_size=int(parameters["document_batch_size"]),
                    max_input_length=int(parameters["max_input_length"]),
                    max_new_tokens=int(parameters["max_new_tokens"]),
                )
                document_tables = run_document_inference(
                    documents,
                    predictor=document_predictor,
                    retriever=retriever,
                    model_id=spec.model_id,
                    task=task,
                    top_k=int(parameters["retrieval_top_k"]),
                )
                document_aggregates.append(document_tables.aggregates)
                document_pairs.append(document_tables.pairs)
                document_errors.append(document_tables.errors)

        scores = concatenate_tables([tables.scores for tables in validation_tables])
        combined_document_pairs = concatenate_tables(document_pairs)
        workbook = write_results_workbook(
            f"ready_llm_{slugify_model_id(spec.model_id)}",
            {
                "scores": scores,
                "per_class": concatenate_tables(
                    [tables.per_class for tables in validation_tables]
                ),
                "confusion_matrix": concatenate_tables(
                    [tables.confusion_matrix for tables in validation_tables]
                ),
                "validation_predictions": concatenate_tables(
                    [tables.predictions for tables in validation_tables]
                ),
                "document_aggregates": concatenate_tables(document_aggregates),
                "document_pairs": combined_document_pairs,
                "errors": concatenate_tables(document_errors),
            },
            {
                "workflow": "ready_llm_evaluation",
                "model_id": spec.model_id,
                "requested_revision": revision or "registry_default",
                "resolved_revision": effective_revision,
                "parameters": parameters,
                "binary_validation_sha256": binary_hash,
                "ternary_validation_sha256": ternary_hash,
                "binary_prompt_sha256": prompt_sha256(prompt_for_task("binary")),
                "ternary_prompt_sha256": prompt_sha256(prompt_for_task("ternary")),
                "rag_commit": rag_commit,
                "compute_dtype": loaded.compute_dtype,
                "summary_enabled": False,
                "reproducibility": reproducibility,
                "bitwise_reproducibility_scope": "same code, lockfile, GPU model, driver, and CUDA stack",
            },
            output_dir=results_dir,
        )
        review_package = write_document_review_package(
            f"ready_llm_{slugify_model_id(spec.model_id)}",
            combined_document_pairs,
            output_dir=results_dir,
        )
        display_scores(scores)
        download_file(workbook)
        if review_package is not None:
            download_file(review_package)
        return scores
    finally:
        del predictors
        del loaded
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass
