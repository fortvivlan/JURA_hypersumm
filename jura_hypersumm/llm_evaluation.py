"""Ready-LLM validation and full RAG document-inference workflow."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

from .colab_support import (
    deliver_file,
    get_huggingface_token,
    selected_docx_files,
)
from .autotest_scoring import (
    add_test_dataset,
    discover_autotest_datasets,
    score_autotest_predictions,
)
from .common import (
    DEFAULT_AUTOTEST_DIR,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_RAG_DIR,
    DEFAULT_RAG_REVISION,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEST_DOCX_DIR,
    EvaluationTables,
    announce_stage,
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
        dataframe["premise"].tolist(),
        dataframe["hypothesis"].tolist(),
        progress_description=f"Validating ready LLM {task}",
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
    document_paths: Sequence[str | Path] | None = None,
    autotest_dir: str | Path = DEFAULT_AUTOTEST_DIR,
    test_docx_dir: str | Path = DEFAULT_TEST_DOCX_DIR,
    score_autotest: bool = True,
    multiple_test: bool = False,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
):
    """Validate one ready LLM and run both tasks on benchmark DOCX files.

    With no explicit ``document_paths``, all matched repository benchmark files
    are used. ``multiple_test=True`` evaluates paired child folders separately.
    The returned table combines validation and autotest scores.
    """
    parameters = merge_parameters(DEFAULT_INFERENCE_PARAMETERS, inference_parameters)
    configure_reproducibility(
        int(parameters["seed"]),
        deterministic=bool(parameters["deterministic"]),
    )
    spec = resolve_model(model_name)
    workflow = f"ready-llm/{spec.alias}"
    announce_stage(workflow, "setup", "Loading binary and ternary validation data.")
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
    announce_stage(workflow, "rag", "Preparing the pinned RAG repository.")
    rag_path, rag_commit = ensure_rag_repository(
        rag_dir, revision=rag_revision
    )
    announce_stage(workflow, "rag", f"RAG repository ready at {rag_commit[:12]}.")
    announce_stage(
        workflow,
        "load",
        f"Loading {spec.model_id} at revision {effective_revision[:12]}.",
    )
    loaded = load_causal_model(
        spec,
        revision=effective_revision,
        token=token,
        quantization=bool(parameters["quantization"]),
        device_map=parameters["device_map"],
        precision=str(parameters["precision"]),
    )
    announce_stage(workflow, "load", "Ready LLM loaded.")
    validation_tables: list[EvaluationTables] = []
    predictors: dict[str, CausalPredictor] = {}
    try:
        for task in ("binary", "ternary"):
            announce_stage(
                workflow,
                "validation",
                f"Starting {task} validation on {len(datasets[task])} examples.",
            )
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
            announce_stage(
                workflow, "validation", f"{task.capitalize()} validation finished."
            )

        announce_stage(workflow, "testing", "Loading the RAG retrieval index.")
        retriever = PremiseRetriever.from_rag_directory(
            rag_path,
            embedding_revision=str(parameters["embedding_revision"]),
            embedding_device=str(parameters["embedding_device"]),
        )
        announce_stage(workflow, "testing", "RAG retrieval index loaded.")
        announce_stage(
            workflow,
            "testing",
            "Document testing is ready. Preparing benchmark or explicit DOCX files.",
        )
        document_aggregates = []
        document_pairs = []
        document_errors = []
        if document_paths is None:
            datasets = discover_autotest_datasets(
                autotest_dir, test_docx_dir, multiple_test=multiple_test
            )
            dataset_inputs = [
                (
                    dataset.name,
                    dataset.autotest_dir,
                    dataset.docx_dir,
                    dataset.documents,
                )
                for dataset in datasets
            ]
        else:
            dataset_inputs = [
                ("default", Path(autotest_dir), Path(test_docx_dir), document_paths)
            ]

        inference_runs = []
        for dataset_name, review_dir, document_dir, paths in dataset_inputs:
            with selected_docx_files(paths) as documents:
                if not documents:
                    announce_stage(
                        workflow,
                        "testing",
                        f"No DOCX files selected for {dataset_name}; testing skipped.",
                    )
                dataset_pairs = []
                for task in ("binary", "ternary"):
                    if documents:
                        announce_stage(
                            workflow,
                            "testing",
                            f"Starting {dataset_name} {task} full RAG inference for "
                            f"{len(documents)} document(s).",
                        )
                    document_predictor = CausalPredictor(
                        loaded.model,
                        loaded.tokenizer,
                        task,
                        batch_size=int(parameters["document_batch_size"]),
                        max_input_length=int(parameters["max_input_length"]),
                        max_new_tokens=int(parameters["max_new_tokens"]),
                    )
                    tables = run_document_inference(
                        documents,
                        predictor=document_predictor,
                        retriever=retriever,
                        model_id=spec.model_id,
                        task=task,
                        top_k=int(parameters["retrieval_top_k"]),
                    )
                    document_aggregates.append(
                        add_test_dataset(tables.aggregates, dataset_name)
                    )
                    labeled_pairs = add_test_dataset(tables.pairs, dataset_name)
                    dataset_pairs.append(labeled_pairs)
                    document_pairs.append(labeled_pairs)
                    document_errors.append(add_test_dataset(tables.errors, dataset_name))
                    if documents:
                        announce_stage(
                            workflow,
                            "testing",
                            f"{dataset_name} {task} document inference finished.",
                        )
            inference_runs.append(
                (
                    dataset_name,
                    review_dir,
                    document_dir,
                    tuple(documents),
                    concatenate_tables(dataset_pairs),
                )
            )

        combined_document_pairs = concatenate_tables(document_pairs)
        autotest_tables = []
        if score_autotest:
            announce_stage(
                workflow, "scoring", "Scoring fresh document pairs against autotest."
            )
            for dataset_name, review_dir, document_dir, documents, pairs in inference_runs:
                if not documents:
                    continue
                for task in ("binary", "ternary"):
                    autotest_tables.append(
                        score_autotest_predictions(
                            pairs,
                            documents,
                            model_id=spec.model_id,
                            task=task,
                            autotest_dir=review_dir,
                            docx_dir=document_dir,
                            test_dataset=dataset_name,
                        )
                    )
        announce_stage(workflow, "results", "Writing score and review artifacts.")
        scores = concatenate_tables(
            [tables.scores for tables in validation_tables]
            + [tables.scores for tables in autotest_tables]
        )
        workbook = write_results_workbook(
            f"ready_llm_{slugify_model_id(spec.model_id)}",
            {
                "scores": scores,
                "per_class": concatenate_tables(
                    [tables.per_class for tables in validation_tables]
                    + [tables.per_class for tables in autotest_tables]
                ),
                "confusion_matrix": concatenate_tables(
                    [tables.confusion_matrix for tables in validation_tables]
                    + [tables.confusion_matrix for tables in autotest_tables]
                ),
                "validation_predictions": concatenate_tables(
                    [tables.predictions for tables in validation_tables]
                ),
                "document_aggregates": concatenate_tables(document_aggregates),
                "document_pairs": combined_document_pairs,
                "errors": concatenate_tables(document_errors),
                "rag_summary": concatenate_tables(
                    [tables.rag_summary for tables in autotest_tables]
                ),
                "autotest_alignment": concatenate_tables(
                    [tables.alignment for tables in autotest_tables]
                ),
                "inferred_gold": concatenate_tables(
                    [tables.inferred_gold for tables in autotest_tables]
                ),
                "autotest_excluded": concatenate_tables(
                    [tables.excluded for tables in autotest_tables]
                ),
                "file_matching": concatenate_tables(
                    [tables.file_matching for tables in autotest_tables]
                ).drop_duplicates() if autotest_tables else None,
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
                "rag_requested_revision": rag_revision,
                "rag_commit": rag_commit,
                "compute_dtype": loaded.compute_dtype,
                "summary_enabled": False,
                "autotest_enabled": score_autotest,
                "multiple_test": multiple_test,
                "test_datasets": [run[0] for run in inference_runs],
                "autotest_dir": Path(autotest_dir),
                "test_docx_dir": Path(test_docx_dir),
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
        deliver_file(workbook)
        if review_package is not None:
            deliver_file(review_package)
        announce_stage(workflow, "complete", "Workflow finished.")
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
