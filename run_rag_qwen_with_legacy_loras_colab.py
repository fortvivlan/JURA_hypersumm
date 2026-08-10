"""Run the RAG-Qwen/legacy-LoRA evaluation with Colab and Drive paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from jura_hypersumm.common import file_sha256
from jura_hypersumm.full_pipeline import run_full_pipeline_evaluation
from jura_hypersumm.inference import SOURCE_PREFIXED_PREMISE_FORMAT
from jura_hypersumm.model_discovery import InferenceModel, resolve_models_source


LEGACY_ADAPTER_DIRS: dict[tuple[str, str], str] = {
    ("meta-llama/Llama-3.1-8B", "ternary"): "meta-llama_Llama-3.1-8B",
    ("meta-llama/Llama-3.1-8B", "binary"): "meta-llama_Llama-3.1-8B_binary",
    ("mistralai/Ministral-8B-Instruct-2410", "ternary"): (
        "mistralai_Ministral-8B-Instruct-2410(1)"
    ),
    ("mistralai/Ministral-8B-Instruct-2410", "binary"): (
        "mistralai_Ministral-8B-Instruct-2410_binary(1)"
    ),
    ("Qwen/Qwen3-8B", "ternary"): "Qwen_Qwen3-8B",
    ("Qwen/Qwen3-8B", "binary"): "Qwen_Qwen3-8B_binary",
    ("t-tech/T-lite-it-2.1", "ternary"): "t-tech_T-lite-it-2.1",
    ("t-tech/T-lite-it-2.1", "binary"): "t-tech_T-lite-it-2.1_binary",
}

MODEL_FAMILIES = ("bert", "lora", "base_llm")
TASKS = ("binary", "ternary")
BASE_MODEL_REVISIONS = {
    "meta-llama/Llama-3.1-8B": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "mistralai/Ministral-8B-Instruct-2410": (
        "2f494a194c5b980dfb9772cb92d26cbb671fce5a"
    ),
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "t-tech/T-lite-it-2.1": "d125c970c553de58fcee3c937d5e4867d4a448d8",
}
TRAINING_PROMPT_SHA256 = {
    "binary": "07f74a37e06e33bd97b07636016a54fd36b05920c7ea41327bc5ec1a39a4c92b",
    "ternary": "0224fb5b30bd6c5034e202618d5c64754d566ead2456ed080140aff03c3da828",
}


def _resolve_path(value: str | Path, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def _adapter_base(adapter_dir: Path) -> str:
    config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    base = str(config.get("base_model_name_or_path") or "").strip()
    if not base:
        raise ValueError(f"Legacy adapter has no base model: {adapter_dir}")
    return base


def _bert_artifacts(bert_models_dir: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for task in TASKS:
        candidates = [
            bert_models_dir / task,
            bert_models_dir / "models" / "bert" / task,
        ]
        for config_path in bert_models_dir.rglob("run_config.json"):
            try:
                run_config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str(run_config.get("task", "")).casefold() == task:
                candidates.append(config_path.parent)
        complete = []
        for candidate in dict.fromkeys(path.resolve() for path in candidates):
            has_weights = any(
                (candidate / filename).is_file()
                for filename in ("model.safetensors", "pytorch_model.bin")
            )
            has_tokenizer = any(
                (candidate / filename).is_file()
                for filename in ("tokenizer.json", "vocab.txt")
            )
            if (candidate / "config.json").is_file() and has_weights and has_tokenizer:
                complete.append(candidate)
        if len(complete) != 1:
            raise FileNotFoundError(
                f"Expected one complete {task} BERT artifact below "
                f"{bert_models_dir}, found {len(complete)}"
            )
        artifacts[task] = complete[0]
    return artifacts


def _mixed_models(
    campaign_dir: Path,
    adapters_dir: Path,
    bert_models_dir: Path,
) -> tuple[
    list[InferenceModel],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    models = resolve_models_source(campaign_dir)
    loras = [model for model in models if model.family == "lora"]
    if len(models) != 18 or len(loras) != 8:
        raise ValueError(
            "Expected the full 18-job campaign matrix with eight LoRAs; "
            f"discovered {len(models)} jobs and {len(loras)} LoRAs in {campaign_dir}"
        )

    replacements: dict[tuple[str, str], Path] = {}
    provenance: dict[str, dict[str, str]] = {}
    for key, directory_name in LEGACY_ADAPTER_DIRS.items():
        adapter_dir = (adapters_dir / directory_name).resolve()
        required = (
            adapter_dir / "adapter_config.json",
            adapter_dir / "adapter_model.safetensors",
            adapter_dir / "tokenizer_config.json",
            adapter_dir / "tokenizer.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Incomplete Drive adapter: " + ", ".join(missing)
            )
        actual_base = _adapter_base(adapter_dir)
        if actual_base != key[0]:
            raise ValueError(
                f"Adapter base mismatch at {adapter_dir}: "
                f"{actual_base!r} != {key[0]!r}"
            )
        replacements[key] = adapter_dir
        provenance[f"{key[0]}:{key[1]}"] = {
            "adapter_dir": str(adapter_dir),
            "adapter_sha256": file_sha256(adapter_dir / "adapter_model.safetensors"),
            "adapter_config_sha256": file_sha256(
                adapter_dir / "adapter_config.json"
            ),
        }

    bert_artifacts = _bert_artifacts(bert_models_dir)
    bert_provenance: dict[str, dict[str, str]] = {}
    for task, artifact_dir in bert_artifacts.items():
        weights_path = next(
            path
            for path in (
                artifact_dir / "model.safetensors",
                artifact_dir / "pytorch_model.bin",
            )
            if path.is_file()
        )
        bert_provenance[task] = {
            "artifact_dir": str(artifact_dir),
            "weights_sha256": file_sha256(weights_path),
            "config_sha256": file_sha256(artifact_dir / "config.json"),
        }

    mixed: list[InferenceModel] = []
    replaced_names: set[str] = set()
    for model in models:
        if model.family == "bert":
            mixed.append(
                replace(model, path_or_id=str(bert_artifacts[model.task]))
            )
            continue
        if model.family != "lora":
            mixed.append(model)
            continue
        key = (str(model.base_model_path_or_id), model.task)
        if key not in replacements:
            raise ValueError(
                f"No Drive adapter matches campaign job {model.name}: {key!r}"
            )
        mixed.append(
            replace(
                model,
                path_or_id=str(replacements[key]),
                training_premise_format=SOURCE_PREFIXED_PREMISE_FORMAT,
            )
        )
        replaced_names.add(model.name)
    if len(replaced_names) != 8:
        raise ValueError("Drive adapter replacement did not resolve eight jobs")
    return mixed, provenance, bert_provenance


def _drive_models(
    adapters_dir: Path,
    bert_models_dir: Path,
) -> tuple[
    list[InferenceModel],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    """Build the complete matrix directly from Drive artifacts."""
    replacements: dict[tuple[str, str], Path] = {}
    adapter_provenance: dict[str, dict[str, str]] = {}
    for key, directory_name in LEGACY_ADAPTER_DIRS.items():
        adapter_dir = (adapters_dir / directory_name).resolve()
        required = (
            adapter_dir / "adapter_config.json",
            adapter_dir / "adapter_model.safetensors",
            adapter_dir / "tokenizer_config.json",
            adapter_dir / "tokenizer.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Incomplete Drive adapter: " + ", ".join(missing)
            )
        actual_base = _adapter_base(adapter_dir)
        if actual_base != key[0]:
            raise ValueError(
                f"Adapter base mismatch at {adapter_dir}: "
                f"{actual_base!r} != {key[0]!r}"
            )
        replacements[key] = adapter_dir
        adapter_provenance[f"{key[0]}:{key[1]}"] = {
            "adapter_dir": str(adapter_dir),
            "adapter_sha256": file_sha256(
                adapter_dir / "adapter_model.safetensors"
            ),
            "adapter_config_sha256": file_sha256(
                adapter_dir / "adapter_config.json"
            ),
        }

    bert_artifacts = _bert_artifacts(bert_models_dir)
    bert_provenance: dict[str, dict[str, str]] = {}
    models: list[InferenceModel] = []
    for task in TASKS:
        artifact_dir = bert_artifacts[task]
        weights_path = next(
            path
            for path in (
                artifact_dir / "model.safetensors",
                artifact_dir / "pytorch_model.bin",
            )
            if path.is_file()
        )
        run_config_path = artifact_dir / "run_config.json"
        run_config = (
            json.loads(run_config_path.read_text(encoding="utf-8"))
            if run_config_path.is_file()
            else {}
        )
        bert_provenance[task] = {
            "artifact_dir": str(artifact_dir),
            "weights_sha256": file_sha256(weights_path),
            "config_sha256": file_sha256(artifact_dir / "config.json"),
        }
        models.append(
            InferenceModel(
                name=f"models__bert__{task}__{task}",
                family="bert",
                task=task,
                path_or_id=str(artifact_dir),
                revision=run_config.get("resolved_revision"),
            )
        )

    for base_model, revision in BASE_MODEL_REVISIONS.items():
        slug = base_model.replace("/", "_")
        for task in TASKS:
            models.append(
                InferenceModel(
                    name=f"models__lora__{slug}__{task}__{task}",
                    family="lora",
                    task=task,
                    path_or_id=str(replacements[(base_model, task)]),
                    revision=revision,
                    base_model_path_or_id=base_model,
                    training_prompt_sha256=TRAINING_PROMPT_SHA256[task],
                    training_premise_format=SOURCE_PREFIXED_PREMISE_FORMAT,
                )
            )
    for base_model, revision in BASE_MODEL_REVISIONS.items():
        slug = base_model.replace("/", "_")
        for task in TASKS:
            models.append(
                InferenceModel(
                    name=f"base__{slug}__{task}",
                    family="base_llm",
                    task=task,
                    path_or_id=base_model,
                    revision=revision,
                )
            )
    if len(models) != 18:
        raise RuntimeError(f"Expected 18 Drive jobs, constructed {len(models)}")
    return models, adapter_provenance, bert_provenance


def _write_models_input(models: Sequence[InferenceModel], path: Path) -> Path:
    entries: list[dict[str, Any]] = []
    for model in models:
        entry: dict[str, Any] = {
            "name": model.name,
            "family": model.family,
            "revision": model.revision,
            "trust_remote_code": model.trust_remote_code,
            "training_prompt_sha256": model.training_prompt_sha256,
            "training_premise_format": model.training_premise_format,
        }
        if model.family == "base_llm":
            entry["tasks"] = [model.task]
        else:
            entry["task"] = model.task
        source = Path(model.path_or_id)
        entry["path" if source.exists() else "model_id"] = (
            str(source.resolve()) if source.exists() else model.path_or_id
        )
        if model.base_model_path_or_id:
            base_source = Path(model.base_model_path_or_id)
            entry[
                "base_model_path" if base_source.exists() else "base_model_id"
            ] = (
                str(base_source.resolve())
                if base_source.exists()
                else model.base_model_path_or_id
            )
        entries.append(entry)
    path.write_text(
        json.dumps({"models": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _normalize_selection(
    values: Sequence[str], allowed: Sequence[str], label: str
) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    unknown = sorted(set(normalized) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    if not normalized:
        raise ValueError(f"At least one {label} is required")
    return normalized


def _preflight_data(
    *,
    project_root: Path,
    data_root: Path,
    autotest_dir: Path,
    test_docx_dir: Path,
    rag_source: Path,
) -> None:
    required_files = (
        project_root / "prompt.py",
        project_root / "prompt_binary.py",
        data_root / "val_binary.csv",
        data_root / "val_ternary.csv",
        rag_source / "rag_manifest.json",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if not autotest_dir.is_dir():
        missing.append(str(autotest_dir))
    if not test_docx_dir.is_dir():
        missing.append(str(test_docx_dir))
    if missing:
        raise FileNotFoundError(
            "Colab evaluation inputs are missing: " + ", ".join(missing)
        )


def run_rag_qwen_with_legacy_loras_colab(
    *,
    project_root: str | Path = "/content/JURA_hypersumm",
    data_root: str | Path = "/content",
    drive_root: str | Path = "/content/drive/MyDrive",
    campaign_dir: str | Path | None = None,
    bert_models_dir: str | Path | None = None,
    lora_adapters_dir: str | Path | None = None,
    rag_source: str | Path | None = None,
    autotest_dir: str | Path | None = None,
    test_docx_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    families: Sequence[str] = MODEL_FAMILIES,
    tasks: Sequence[str] = TASKS,
    document_batch_size: int = 4,
    llm_batch_size: int = 1,
    embedding_device: str = "cuda",
    reranker_device: str = "cuda",
    reranker_batch_size: int = 8,
    reranker_precision: str = "auto",
    candidate_top_k: int = 100,
    final_top_k: int = 60,
    quantization: bool = True,
    precision: str = "auto",
    device_map: str = "auto",
    max_retries: int = 0,
    dry_run: bool = False,
):
    """Evaluate selected jobs using content data and Drive model artifacts.

    The function expects validation CSVs and benchmark folders under
    ``data_root`` by default. It reads legacy adapters and the portable
    ``rag-qwen`` bundle from mounted Drive and writes resumable results back to
    Drive. It does not mount Drive or install packages.
    """
    project = Path(project_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    drive = Path(drive_root).expanduser().resolve()
    if not drive.is_dir():
        raise FileNotFoundError(
            f"Drive root is unavailable; mount Drive before calling: {drive}"
        )

    campaign = (
        _resolve_path(campaign_dir, relative_to=data)
        if campaign_dir is not None
        else None
    )
    bert_models = _resolve_path(
        bert_models_dir or "croc_bert", relative_to=drive
    )
    adapters = _resolve_path(
        lora_adapters_dir or "lora_adapters", relative_to=drive
    )
    rag = _resolve_path(rag_source or "rag-qwen", relative_to=drive)
    autotest = _resolve_path(autotest_dir or "autotest", relative_to=data)
    documents = _resolve_path(test_docx_dir or "test_docx", relative_to=data)
    output = _resolve_path(
        results_dir
        or "JURA_hypersumm_results/full_pipeline_evaluation_rag_qwen_legacy_lora",
        relative_to=drive,
    )

    selected_families = _normalize_selection(
        families, MODEL_FAMILIES, "model families"
    )
    selected_tasks = _normalize_selection(tasks, TASKS, "tasks")
    if candidate_top_k <= 0 or final_top_k <= 0:
        raise ValueError("candidate_top_k and final_top_k must be positive")
    if final_top_k > candidate_top_k:
        raise ValueError("final_top_k cannot exceed candidate_top_k")
    _preflight_data(
        project_root=project,
        data_root=data,
        autotest_dir=autotest,
        test_docx_dir=documents,
        rag_source=rag,
    )
    if campaign is None:
        models, provenance, bert_provenance = _drive_models(
            adapters, bert_models
        )
    else:
        models, provenance, bert_provenance = _mixed_models(
            campaign, adapters, bert_models
        )
    selected_models = [
        model
        for model in models
        if model.family in selected_families and model.task in selected_tasks
    ]
    if not selected_models:
        raise ValueError("The family/task selection produced no evaluation jobs")

    adapter_digest = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode("utf-8")
    ).hexdigest()
    artifact_digest = hashlib.sha256(
        json.dumps(
            {"legacy_adapters": provenance, "bert_models": bert_provenance},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    plan = {
        "project_root": str(project),
        "data_root": str(data),
        "campaign_dir": str(campaign) if campaign is not None else None,
        "bert_models_dir": str(bert_models),
        "lora_adapters_dir": str(adapters),
        "rag_source": str(rag),
        "autotest_dir": str(autotest),
        "test_docx_dir": str(documents),
        "results_dir": str(output),
        "adapter_set_sha256": adapter_digest,
        "artifact_set_sha256": artifact_digest,
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "models": [asdict(model) for model in selected_models],
        "legacy_adapters": provenance,
        "bert_models": bert_provenance,
    }
    if dry_run:
        return plan

    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "legacy_adapter_set.json"
    if provenance_path.is_file():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing.get("artifact_set_sha256") != artifact_digest:
            raise ValueError(
                "The Drive results directory belongs to another adapter set; "
                "choose a new results_dir."
            )
    provenance_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    models_path = _write_models_input(
        selected_models, output / "colab_models_input.json"
    )
    return run_full_pipeline_evaluation(
        models_source=models_path,
        rag_source=rag,
        prompt_set="base",
        reranker_mode="bundle",
        repo_root=project,
        validation_dir=data,
        autotest_dir=autotest,
        test_docx_dir=documents,
        results_dir=output,
        inference_parameters={
            "document_batch_size": document_batch_size,
            "llm_batch_size": llm_batch_size,
            "embedding_device": embedding_device,
            "candidate_top_k": candidate_top_k,
            "final_top_k": final_top_k,
            "reranker_device": reranker_device,
            "reranker_batch_size": reranker_batch_size,
            "reranker_precision": reranker_precision,
            "quantization": quantization,
            "precision": precision,
            "device_map": device_map,
            "max_retries": max_retries,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/content/JURA_hypersumm"))
    parser.add_argument("--data-root", type=Path, default=Path("/content"))
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive"))
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--bert-models-dir", type=Path)
    parser.add_argument("--lora-adapters-dir", type=Path)
    parser.add_argument("--rag-source", type=Path)
    parser.add_argument("--autotest-dir", type=Path)
    parser.add_argument("--test-docx-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--families", nargs="+", choices=MODEL_FAMILIES, default=MODEL_FAMILIES)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--document-batch-size", type=int, default=4)
    parser.add_argument("--llm-batch-size", type=int, default=1)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--final-top-k", type=int, default=60)
    parser.add_argument(
        "--reranker-precision",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--quantization", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    result = run_rag_qwen_with_legacy_loras_colab(
        project_root=arguments.project_root,
        data_root=arguments.data_root,
        drive_root=arguments.drive_root,
        campaign_dir=arguments.campaign_dir,
        bert_models_dir=arguments.bert_models_dir,
        lora_adapters_dir=arguments.lora_adapters_dir,
        rag_source=arguments.rag_source,
        autotest_dir=arguments.autotest_dir,
        test_docx_dir=arguments.test_docx_dir,
        results_dir=arguments.results_dir,
        families=arguments.families,
        tasks=arguments.tasks,
        document_batch_size=arguments.document_batch_size,
        llm_batch_size=arguments.llm_batch_size,
        embedding_device=arguments.embedding_device,
        reranker_device=arguments.reranker_device,
        reranker_batch_size=arguments.reranker_batch_size,
        reranker_precision=arguments.reranker_precision,
        candidate_top_k=arguments.candidate_top_k,
        final_top_k=arguments.final_top_k,
        quantization=arguments.quantization,
        precision=arguments.precision,
        device_map=arguments.device_map,
        max_retries=arguments.max_retries,
        dry_run=arguments.dry_run,
    )
    if hasattr(result, "to_string"):
        print(result.to_string(index=False))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
