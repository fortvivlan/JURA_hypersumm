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
    (data / "autotest").mkdir()
    (data / "test_docx").mkdir()
    (data / "full_pipeline_v1").mkdir()
    (drive / "lora_adapters").mkdir()
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
        "_mixed_models",
        lambda campaign, adapters: (_models(), {"set": {"adapter_sha256": "a"}}),
    )

    plan = colab_runner.run_rag_qwen_with_legacy_loras_colab(
        project_root=project,
        data_root=data,
        drive_root=drive,
        families=("lora",),
        tasks=("ternary",),
        dry_run=True,
    )

    assert plan["campaign_dir"] == str((data / "full_pipeline_v1").resolve())
    assert plan["lora_adapters_dir"] == str((drive / "lora_adapters").resolve())
    assert plan["rag_source"] == str((drive / "rag-qwen").resolve())
    assert [model["name"] for model in plan["models"]] == ["qwen-lora"]


def test_colab_runner_writes_drive_manifest_and_forwards_data_paths(
    monkeypatch, tmp_path: Path
) -> None:
    project, data, drive = _layout(tmp_path)
    monkeypatch.setattr(
        colab_runner,
        "_mixed_models",
        lambda campaign, adapters: (_models(), {"set": {"adapter_sha256": "a"}}),
    )
    observed = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return pd.DataFrame([{"score": 1.0}])

    monkeypatch.setattr(colab_runner, "run_full_pipeline_evaluation", fake_run)
    output = drive / "results"

    result = colab_runner.run_rag_qwen_with_legacy_loras_colab(
        project_root=project,
        data_root=data,
        drive_root=drive,
        results_dir=output,
        max_retries=0,
    )

    assert result["score"].item() == 1.0
    assert observed["repo_root"] == project.resolve()
    assert observed["validation_dir"] == data.resolve()
    assert observed["autotest_dir"] == (data / "autotest").resolve()
    assert observed["test_docx_dir"] == (data / "test_docx").resolve()
    assert observed["inference_parameters"]["max_retries"] == 0
    assert (output / "legacy_adapter_set.json").is_file()
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
