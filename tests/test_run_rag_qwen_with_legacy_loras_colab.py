import json
from pathlib import Path

import pandas as pd
import pytest

import run_rag_qwen_with_legacy_loras_colab as colab_runner
from jura_hypersumm.model_discovery import InferenceModel


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    data = tmp_path / "content"
    drive = tmp_path / "drive" / "MyDrive"
    project.mkdir()
    data.mkdir()
    drive.mkdir(parents=True)
    (project / "prompt.py").write_text('PROMPT_TEXT = "t"', encoding="utf-8")
    (project / "prompt_binary.py").write_text(
        'PROMPT_TEXT_BIN = "b"', encoding="utf-8"
    )
    for task in ("binary", "ternary"):
        (data / f"val_{task}.csv").write_text("premise,hypothesis,source,tag\n", encoding="utf-8")
    (data / "full_pipeline_v1").mkdir()
    (drive / "jura" / "autotest").mkdir(parents=True)
    (drive / "jura" / "test_docx").mkdir()
    (drive / "lora_adapters").mkdir()
    (drive / "croc_bert").mkdir()
    rag = drive / "rag-qwen"
    rag.mkdir()
    (rag / "rag_manifest.json").write_text("{}", encoding="utf-8")
    return project, data, drive


def _models() -> list[InferenceModel]:
    return [
        InferenceModel("bert-binary", "bert", "binary", "/models/bert"),
        InferenceModel("qwen-lora", "lora", "ternary", "/drive/qwen"),
        InferenceModel("qwen-base", "base_llm", "ternary", "Qwen/Qwen3-8B"),
    ]


def test_colab_dry_run_resolves_content_and_drive_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    project, data, drive = _layout(tmp_path)
    monkeypatch.setattr(
        colab_runner,
        "_drive_models",
        lambda adapters, bert: (
            _models(),
            {"set": {"adapter_sha256": "a"}},
            {"binary": {"weights_sha256": "b"}},
        ),
    )

    plan = colab_runner.run_rag_qwen_with_legacy_loras_colab(
        project_root=project,
        data_root=data,
        drive_root=drive,
        families=("lora",),
        tasks=("ternary",),
        dry_run=True,
    )

    assert plan["campaign_dir"] is None
    assert plan["bert_models_dir"] == str((drive / "croc_bert").resolve())
    assert plan["lora_adapters_dir"] == str((drive / "lora_adapters").resolve())
    assert plan["rag_source"] == str((drive / "rag-qwen").resolve())
    assert plan["autotest_dir"] == str((drive / "jura" / "autotest").resolve())
    assert plan["test_docx_dir"] == str((drive / "jura" / "test_docx").resolve())
    assert [model["name"] for model in plan["models"]] == ["qwen-lora"]


def test_colab_runner_writes_drive_manifest_and_forwards_data_paths(
    monkeypatch, tmp_path: Path
) -> None:
    project, data, drive = _layout(tmp_path)
    monkeypatch.setattr(
        colab_runner,
        "_drive_models",
        lambda adapters, bert: (
            _models(),
            {"set": {"adapter_sha256": "a"}},
            {"binary": {"weights_sha256": "b"}},
        ),
    )
    observed = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return pd.DataFrame([{"score": 1.0}])

    monkeypatch.setattr(colab_runner, "run_full_pipeline_evaluation", fake_run)
    output = drive / "results"
    output.mkdir()
    (output / "legacy_adapter_set.json").write_text("", encoding="utf-8")

    result = colab_runner.run_rag_qwen_with_legacy_loras_colab(
        project_root=project,
        data_root=data,
        drive_root=drive,
        results_dir=output,
        candidate_top_k=40,
        final_top_k=15,
        max_retries=0,
    )

    assert result["score"].item() == 1.0
    assert observed["repo_root"] == project.resolve()
    assert observed["validation_dir"] == data.resolve()
    assert observed["autotest_dir"] == (drive / "jura" / "autotest").resolve()
    assert observed["test_docx_dir"] == (drive / "jura" / "test_docx").resolve()
    assert observed["inference_parameters"]["max_retries"] == 0
    assert observed["inference_parameters"]["candidate_top_k"] == 40
    assert observed["inference_parameters"]["final_top_k"] == 15
    assert observed["hf_cache_evictions"] == {}
    assert observed["job_execution_order"] == (
        "bert-binary",
        "qwen-lora",
        "qwen-base",
    )
    assert (output / "legacy_adapter_set.json").is_file()
    assert list(output.glob("legacy_adapter_set.corrupt-*.json"))
    models = json.loads((output / "colab_models_input.json").read_text(encoding="utf-8"))
    assert len(models["models"]) == 3


def test_colab_runner_requires_mounted_drive(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="mount Drive"):
        colab_runner.run_rag_qwen_with_legacy_loras_colab(
            project_root=tmp_path,
            data_root=tmp_path,
            drive_root=tmp_path / "missing-drive",
            dry_run=True,
        )


def test_bert_artifacts_support_direct_binary_and_ternary_folders(
    tmp_path: Path,
) -> None:
    root = tmp_path / "croc_bert"
    for task in ("binary", "ternary"):
        artifact = root / task
        artifact.mkdir(parents=True)
        (artifact / "config.json").write_text("{}", encoding="utf-8")
        (artifact / "model.safetensors").write_bytes(b"weights")
        (artifact / "tokenizer.json").write_text("{}", encoding="utf-8")

    artifacts = colab_runner._bert_artifacts(root)

    assert artifacts == {
        "binary": (root / "binary").resolve(),
        "ternary": (root / "ternary").resolve(),
    }


def test_drive_models_constructs_full_matrix_without_campaign(
    tmp_path: Path,
) -> None:
    adapters = tmp_path / "lora_adapters"
    for (base_model, _task), directory_name in colab_runner.LEGACY_ADAPTER_DIRS.items():
        artifact = adapters / directory_name
        artifact.mkdir(parents=True)
        (artifact / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": base_model}), encoding="utf-8"
        )
        (artifact / "adapter_model.safetensors").write_bytes(b"adapter")
        (artifact / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (artifact / "tokenizer.json").write_text("{}", encoding="utf-8")
    bert = tmp_path / "croc_bert"
    for task in ("binary", "ternary"):
        artifact = bert / task
        artifact.mkdir(parents=True)
        (artifact / "config.json").write_text("{}", encoding="utf-8")
        (artifact / "model.safetensors").write_bytes(b"bert")
        (artifact / "tokenizer.json").write_text("{}", encoding="utf-8")

    models, adapter_provenance, bert_provenance = colab_runner._drive_models(
        adapters, bert
    )

    assert len(models) == 18
    assert sum(model.family == "bert" for model in models) == 2
    assert sum(model.family == "lora" for model in models) == 8
    assert sum(model.family == "base_llm" for model in models) == 8
    assert len(adapter_provenance) == 8
    assert set(bert_provenance) == {"binary", "ternary"}


def test_colab_runner_rejects_final_depth_above_candidate_depth(
    monkeypatch, tmp_path: Path
) -> None:
    project, data, drive = _layout(tmp_path)

    with pytest.raises(ValueError, match="cannot exceed"):
        colab_runner.run_rag_qwen_with_legacy_loras_colab(
            project_root=project,
            data_root=data,
            drive_root=drive,
            candidate_top_k=10,
            final_top_k=20,
            dry_run=True,
        )


def test_llama_cache_boundaries_and_execution_order_are_targeted() -> None:
    revision = colab_runner.BASE_MODEL_REVISIONS[colab_runner.LLAMA_MODEL_ID]
    models = [
        InferenceModel("bert", "bert", "binary", "/bert"),
        InferenceModel(
            "llama-lora-binary",
            "lora",
            "binary",
            "/llama-adapter-binary",
            base_model_path_or_id=colab_runner.LLAMA_MODEL_ID,
        ),
        InferenceModel(
            "llama-lora-ternary",
            "lora",
            "ternary",
            "/llama-adapter-ternary",
            base_model_path_or_id=colab_runner.LLAMA_MODEL_ID,
        ),
        InferenceModel(
            "qwen-lora", "lora", "binary", "/qwen-adapter",
            base_model_path_or_id="Qwen/Qwen3-8B",
        ),
        InferenceModel(
            "llama-base-binary",
            "base_llm",
            "binary",
            colab_runner.LLAMA_MODEL_ID,
        ),
        InferenceModel(
            "llama-base-ternary",
            "base_llm",
            "ternary",
            colab_runner.LLAMA_MODEL_ID,
        ),
        InferenceModel("qwen-base", "base_llm", "binary", "Qwen/Qwen3-8B"),
    ]

    assert colab_runner._llama_cache_evictions(models) == {
        "llama-lora-ternary": (colab_runner.LLAMA_MODEL_ID, revision),
        "llama-base-ternary": (colab_runner.LLAMA_MODEL_ID, revision),
    }
    assert colab_runner._disk_efficient_execution_order(models) == (
        "bert",
        "llama-lora-binary",
        "llama-lora-ternary",
        "llama-base-binary",
        "llama-base-ternary",
        "qwen-lora",
        "qwen-base",
    )
