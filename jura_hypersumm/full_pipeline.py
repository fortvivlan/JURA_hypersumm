"""Inference-only evaluation across discovered BERT, base LLM, and LoRA models."""

from __future__ import annotations

import gc
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any, Mapping

from .autotest_scoring import discover_autotest_datasets, score_autotest_predictions
from .bert import BertPredictor, _load_bert_artifact
from .colab_support import get_huggingface_token
from .common import (
    ModelSpec,
    evaluate_predictions,
    load_dataset,
    merge_parameters,
    resolve_huggingface_revision,
)
from .inference import (
    BODY_ONLY_PREMISE_FORMAT,
    SOURCE_PREFIXED_PREMISE_FORMAT,
    format_model_premise,
    run_document_inference,
)
from .llm_common import CausalPredictor, load_causal_model
from .model_discovery import InferenceModel, resolve_models_source, write_resolved_models
from .prompt_sets import load_prompt_set
from .rag.artifacts import load_rag_bundle
from .rag.reranking import CrossEncoderReranker
from .retrieval import PremiseRetriever

DEFAULT_FULL_PIPELINE_PARAMETERS: dict[str, Any] = {
    "bert_batch_size": 32,
    "bert_max_length": 512,
    "llm_batch_size": 1,
    "document_batch_size": 4,
    "max_input_length": 4096,
    "max_new_tokens": 16,
    "quantization": True,
    "device_map": "auto",
    "precision": "auto",
    "embedding_device": "cpu",
    "retrieval_top_k": 20,
    "candidate_top_k": 20,
    "final_top_k": 20,
    "reranker_batch_size": 8,
    "reranker_max_length": 1024,
    "reranker_precision": "auto",
    "reranker_device": "cuda",
    "max_retries": 1,
}


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _include_source_prefix(model_family: str) -> bool:
    """Keep BERT body-only while exposing provision headings to generative models."""
    return model_family != "bert"


def _load_adapter_tokenizer(
    adapter_dir: Path,
    *,
    base_tokenizer,
    trust_remote_code: bool,
):
    """Load an adapter tokenizer, falling back across Transformers versions.

    PEFT adapters do not change vocabulary. Some legacy ``tokenizer_config``
    files reference tokenizer classes or special-token schemas introduced by a
    newer Transformers release. In that case, the already loaded tokenizer
    from the adapter's pinned base model is the compatible equivalent.
    """
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    except (AttributeError, TypeError, ValueError):
        tokenizer = base_tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_llm(entry: InferenceModel, parameters: Mapping[str, Any]):
    from peft import PeftModel

    token = get_huggingface_token()
    if entry.family == "lora":
        adapter_dir = Path(entry.path_or_id)
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        base_source = entry.base_model_path_or_id or adapter_config.get("base_model_name_or_path")
        if not base_source:
            raise ValueError(f"No base model configured for {entry.name}")
    else:
        base_source = entry.path_or_id
    local = Path(base_source).exists()
    revision = None if local else resolve_huggingface_revision(
        base_source, entry.revision, token=token
    )
    spec = ModelSpec(entry.name, str(base_source), revision, entry.trust_remote_code)
    loaded = load_causal_model(
        spec,
        revision=revision,
        token=token,
        quantization=bool(parameters["quantization"]),
        device_map=parameters["device_map"],
        precision=str(parameters["precision"]),
    )
    if entry.family == "lora":
        tokenizer = _load_adapter_tokenizer(
            Path(entry.path_or_id),
            base_tokenizer=loaded.tokenizer,
            trust_remote_code=entry.trust_remote_code,
        )
        model = PeftModel.from_pretrained(
            loaded.model,
            entry.path_or_id,
            local_files_only=True,
            is_trainable=False,
        )
        model.eval()
        return model, tokenizer
    return loaded.model, loaded.tokenizer


def _load_predictor(entry: InferenceModel, prompt_text: str, parameters):
    if entry.family == "bert":
        model, tokenizer, _ = _load_bert_artifact(
            Path(entry.path_or_id),
            task=entry.task,  # type: ignore[arg-type]
            parameters={"device": "auto"},
        )
        return model, tokenizer, BertPredictor(
            model,
            tokenizer,
            batch_size=int(parameters["bert_batch_size"]),
            max_length=int(parameters["bert_max_length"]),
        )
    model, tokenizer = _load_llm(entry, parameters)
    return model, tokenizer, CausalPredictor(
        model,
        tokenizer,
        entry.task,  # type: ignore[arg-type]
        batch_size=int(parameters["llm_batch_size"]),
        max_input_length=int(parameters["max_input_length"]),
        max_new_tokens=int(parameters["max_new_tokens"]),
        prompt_text=prompt_text,
    )


def _write_job_tables(job_dir: Path, tables: dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    with __import__("pandas").ExcelWriter(job_dir / "results.xlsx", engine="openpyxl") as writer:
        for name, table in tables.items():
            if table is not None:
                table.to_excel(writer, sheet_name=name[:31], index=False)
    tables["scores"].to_csv(job_dir / "scores.csv", index=False, encoding="utf-8")


def _evaluate_one(
    entry: InferenceModel,
    *,
    predictor,
    retriever,
    val_path: Path,
    autotest_dir: Path,
    test_docx_dir: Path,
    parameters,
):
    import pandas as pd

    validation = load_dataset(val_path, entry.task)  # type: ignore[arg-type]
    include_source_prefix = _include_source_prefix(entry.family)
    validation_premises = (
        [
            format_model_premise(row.premise, row.source)
            for row in validation.itertuples(index=False)
        ]
        if include_source_prefix
        else validation["premise"].tolist()
    )
    predictions = predictor.predict_examples(
        validation_premises,
        validation["hypothesis"].tolist(),
        progress_description=f"Validating {entry.name}",
    )
    evaluation = evaluate_predictions(
        validation,
        [prediction.label for prediction in predictions],
        [prediction.raw_output for prediction in predictions],
        model_id=entry.name,
        task=entry.task,  # type: ignore[arg-type]
    )
    evaluation.predictions["model_premise"] = validation_premises
    all_pairs = []
    all_aggregates = []
    all_errors = []
    autotest_scores = []
    autotest_per_class = []
    autotest_confusions = []
    rag_summaries = []
    alignments = []
    inferred_gold = []
    excluded = []
    file_matching = []
    datasets = discover_autotest_datasets(autotest_dir, test_docx_dir, multiple_test=True)
    for dataset in datasets:
        document_predictor = predictor
        if isinstance(predictor, CausalPredictor):
            document_predictor = CausalPredictor(
                predictor.model,
                predictor.tokenizer,
                predictor.task,
                batch_size=int(parameters["document_batch_size"]),
                max_input_length=int(parameters["max_input_length"]),
                max_new_tokens=int(parameters["max_new_tokens"]),
                prompt_text=predictor.prompt_text,
            )
        inference = run_document_inference(
            dataset.documents,
            predictor=document_predictor,
            retriever=retriever,
            model_id=entry.name,
            task=entry.task,  # type: ignore[arg-type]
            top_k=int(parameters["retrieval_top_k"]),
            final_top_k=int(parameters["final_top_k"]),
            include_source_prefix=include_source_prefix,
        )
        for table in (inference.pairs, inference.aggregates, inference.errors):
            table.insert(0, "test_dataset", dataset.name)
        all_pairs.append(inference.pairs)
        all_aggregates.append(inference.aggregates)
        all_errors.append(inference.errors)
        scored = score_autotest_predictions(
            inference.pairs,
            dataset.documents,
            model_id=entry.name,
            task=entry.task,  # type: ignore[arg-type]
            autotest_dir=dataset.autotest_dir,
            docx_dir=dataset.docx_dir,
            test_dataset=dataset.name,
        )
        autotest_scores.append(scored.scores)
        autotest_per_class.append(scored.per_class)
        autotest_confusions.append(scored.confusion_matrix)
        rag_summaries.append(scored.rag_summary)
        alignments.append(scored.alignment)
        inferred_gold.append(scored.inferred_gold)
        excluded.append(scored.excluded)
        file_matching.append(scored.file_matching)
    scores = pd.concat([evaluation.scores, *autotest_scores], ignore_index=True)
    return {
        "scores": scores,
        "per_class": pd.concat([evaluation.per_class, *autotest_per_class], ignore_index=True),
        "confusion": pd.concat(
            [evaluation.confusion_matrix, *autotest_confusions], ignore_index=True
        ),
        "validation_predictions": evaluation.predictions,
        "document_pairs": pd.concat(all_pairs, ignore_index=True),
        "document_aggregates": pd.concat(all_aggregates, ignore_index=True),
        "errors": pd.concat(all_errors, ignore_index=True),
        "rag_summary": pd.concat(rag_summaries, ignore_index=True),
        "autotest_alignment": pd.concat(alignments, ignore_index=True),
        "inferred_gold": pd.concat(inferred_gold, ignore_index=True),
        "autotest_excluded": pd.concat(excluded, ignore_index=True),
        "file_matching": pd.concat(file_matching, ignore_index=True),
    }


def _cleanup(*objects) -> None:
    del objects
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


def run_full_pipeline_evaluation(
    *,
    models_source: str | Path = "local_artifacts/campaigns/full_pipeline_v1",
    rag_source: str | Path = "dms-rag",
    prompt_set: str = "base",
    reranker_mode: str = "none",
    reranker_model_id: str | None = None,
    reranker_revision: str | None = None,
    reranker_trust_remote_code: bool | None = None,
    repo_root: str | Path = ".",
    validation_dir: str | Path | None = None,
    autotest_dir: str | Path = "autotest",
    test_docx_dir: str | Path = "test_docx",
    results_dir: str | Path = "local_results/full_pipeline_evaluation",
    inference_parameters: Mapping[str, Any] | None = None,
):
    """Re-evaluate every selected model on validation and full DOCX benchmarks."""
    import pandas as pd

    overrides = dict(inference_parameters or {})
    parameters = merge_parameters(DEFAULT_FULL_PIPELINE_PARAMETERS, overrides)
    if "candidate_top_k" not in overrides and "retrieval_top_k" in overrides:
        parameters["candidate_top_k"] = int(overrides["retrieval_top_k"])
        if "final_top_k" not in overrides:
            parameters["final_top_k"] = int(overrides["retrieval_top_k"])
    parameters["retrieval_top_k"] = int(parameters["candidate_top_k"])
    if int(parameters["candidate_top_k"]) <= 0 or int(parameters["final_top_k"]) <= 0:
        raise ValueError("candidate_top_k and final_top_k must be positive")
    if int(parameters["final_top_k"]) > int(parameters["candidate_top_k"]):
        raise ValueError("final_top_k cannot exceed candidate_top_k")
    root = Path(repo_root).resolve()
    validation_root = (
        root
        if validation_dir is None
        else Path(validation_dir).expanduser().resolve()
    )
    source_path = Path(models_source)
    if not source_path.is_absolute():
        source_path = root / source_path
    models = resolve_models_source(source_path)
    prompts = load_prompt_set(prompt_set, root=root)
    rag_path = Path(rag_source)
    if not rag_path.is_absolute():
        rag_path = root / rag_path
    bundle = load_rag_bundle(rag_path)
    normalized_reranker_mode = reranker_mode.strip().lower()
    if normalized_reranker_mode not in {"none", "bundle", "pretrained"}:
        raise ValueError("reranker_mode must be none, bundle, or pretrained")
    reranker = None
    reranker_config = None
    if normalized_reranker_mode != "none":
        if normalized_reranker_mode == "bundle":
            if bundle.reranker is None:
                raise ValueError("Selected RAG bundle does not define a reranker")
            selected_model = bundle.reranker.model
            selected_revision = bundle.reranker.revision
            selected_trust = bundle.reranker.trust_remote_code
            selected_max_length = bundle.reranker.max_length
        else:
            selected_model = reranker_model_id or "Alibaba-NLP/gte-multilingual-reranker-base"
            selected_revision = reranker_revision
            selected_trust = (
                selected_model == "Alibaba-NLP/gte-multilingual-reranker-base"
                if reranker_trust_remote_code is None
                else bool(reranker_trust_remote_code)
            )
            selected_max_length = int(parameters["reranker_max_length"])
        local_reranker = Path(selected_model).exists()
        if not local_reranker:
            selected_revision = resolve_huggingface_revision(
                selected_model,
                selected_revision,
                token=get_huggingface_token(),
            )
        else:
            selected_revision = None
        reranker = CrossEncoderReranker(
            selected_model,
            revision=selected_revision,
            token=get_huggingface_token(),
            trust_remote_code=selected_trust,
            device=str(parameters["reranker_device"]),
            precision=str(parameters["reranker_precision"]),
            batch_size=int(parameters["reranker_batch_size"]),
            max_length=selected_max_length,
        )
        reranker_config = {
            "mode": normalized_reranker_mode,
            "model": selected_model,
            "revision": selected_revision,
            "trust_remote_code": selected_trust,
            "max_length": selected_max_length,
        }
    output = Path(results_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    write_resolved_models(models, output / "resolved_models.json")
    config = {
        "models": [model.__dict__ for model in models],
        "rag": str(rag_path.resolve()),
        "prompt_set": prompt_set,
        "prompt_hashes": {"binary": prompts.binary_sha256, "ternary": prompts.ternary_sha256},
        "parameters": parameters,
        "reranker": reranker_config,
        "premise_formats": {
            "bert": BODY_ONLY_PREMISE_FORMAT,
            "base_llm": SOURCE_PREFIXED_PREMISE_FORMAT,
            "lora": SOURCE_PREFIXED_PREMISE_FORMAT,
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    state_path = output / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise ValueError("Existing result state belongs to a different evaluation configuration")
    else:
        state = {"fingerprint": fingerprint, "jobs": {}}
    retriever = PremiseRetriever.from_components(
        bundle.codex_path,
        bundle.index_dir,
        embedding_model=bundle.embedding_model,
        embedding_revision=bundle.embedding_revision,
        embedding_device=str(parameters["embedding_device"]),
        normalize_embeddings=bundle.normalize_embeddings,
        embedding_query_prefix=bundle.embedding_query_prefix,
        embedding_document_prefix=bundle.embedding_document_prefix,
        embedding_trust_remote_code=bundle.embedding_trust_remote_code,
        embedding_precision=bundle.embedding_precision,
        embedding_batch_size=bundle.embedding_batch_size,
        reranker=reranker,
    )
    score_frames = []
    for entry in models:
        job_dir = output / "jobs" / _safe_name(entry.name)
        score_path = job_dir / "scores.csv"
        record = state["jobs"].setdefault(entry.name, {"status": "pending", "attempts": 0})
        if record["status"] == "completed" and score_path.is_file():
            score_frames.append(pd.read_csv(score_path))
            continue
        error = None
        for _ in range(int(parameters["max_retries"]) + 1):
            record["attempts"] += 1
            record["status"] = "running"
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            model = tokenizer = predictor = None
            try:
                prompt_text = prompts.binary if entry.task == "binary" else prompts.ternary
                model, tokenizer, predictor = _load_predictor(entry, prompt_text, parameters)
                tables = _evaluate_one(
                    entry,
                    predictor=predictor,
                    retriever=retriever,
                    val_path=validation_root / f"val_{entry.task}.csv",
                    autotest_dir=Path(autotest_dir) if Path(autotest_dir).is_absolute() else root / autotest_dir,
                    test_docx_dir=Path(test_docx_dir) if Path(test_docx_dir).is_absolute() else root / test_docx_dir,
                    parameters=parameters,
                )
                inference_hash = prompts.binary_sha256 if entry.task == "binary" else prompts.ternary_sha256
                scores = tables["scores"].copy()
                scores.insert(0, "model_name", entry.name)
                scores.insert(1, "model_family", entry.family)
                scores.insert(2, "rag_version", bundle.name)
                scores.insert(3, "reranker_mode", normalized_reranker_mode)
                scores.insert(4, "reranker_model", (
                    reranker_config["model"] if reranker_config else "none"
                ))
                scores.insert(5, "candidate_top_k", parameters["candidate_top_k"])
                scores.insert(6, "final_top_k", parameters["final_top_k"])
                scores.insert(7, "prompt_set", prompt_set if entry.family != "bert" else "not_applicable")
                scores.insert(8, "prompt_matches_training", (
                    None if entry.family != "lora" or not entry.training_prompt_sha256
                    else entry.training_prompt_sha256 == inference_hash
                ))
                scores.insert(9, "premise_format", (
                    BODY_ONLY_PREMISE_FORMAT
                    if not _include_source_prefix(entry.family)
                    else SOURCE_PREFIXED_PREMISE_FORMAT
                ))
                scores.insert(10, "premise_matches_training", (
                    None
                    if entry.family != "lora" or not entry.training_premise_format
                    else entry.training_premise_format
                    == SOURCE_PREFIXED_PREMISE_FORMAT
                ))
                tables["scores"] = scores
                _write_job_tables(job_dir, tables)
                score_frames.append(scores)
                record.update({"status": "completed", "error": ""})
                error = None
                break
            except Exception as exc:
                error = exc
                record.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()[-30000:]})
            finally:
                model = tokenizer = predictor = None
                _cleanup()
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        if error is not None:
            continue
    combined = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    combined.to_csv(output / "all_scores.csv", index=False, encoding="utf-8")
    with pd.ExcelWriter(output / "all_scores.xlsx", engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="scores", index=False)
        pd.DataFrame([{"job": key, **value} for key, value in state["jobs"].items()]).to_excel(
            writer, sheet_name="job_status", index=False
        )
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [name for name, value in state["jobs"].items() if value["status"] != "completed"]
    if failed:
        raise RuntimeError("Some evaluation jobs failed; rerun to resume: " + ", ".join(failed))
    return combined
