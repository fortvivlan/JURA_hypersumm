import json
from pathlib import Path

from jura_hypersumm.model_discovery import discover_models, resolve_models_source


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _bert(root: Path, name: str, task: str) -> None:
    path = root / name
    _write_json(path / "config.json", {"architectures": ["BertForSequenceClassification"]})
    _write_json(path / "run_config.json", {"task": task, "resolved_revision": "bert-rev"})
    (path / "model.safetensors").write_bytes(b"model")


def _lora(root: Path, name: str, task: str, base: str) -> None:
    path = root / name
    _write_json(path / "adapter_config.json", {"base_model_name_or_path": base})
    _write_json(
        path / "run_config.json",
        {"task": task, "resolved_revision": "base-rev", "prompt_sha256": "prompt"},
    )
    (path / "adapter_model.safetensors").write_bytes(b"adapter")


def test_directory_discovery_keeps_both_bert_tasks_and_infers_base(tmp_path: Path) -> None:
    _bert(tmp_path, "bert/a", "binary")
    _bert(tmp_path, "bert/b", "ternary")
    _lora(tmp_path, "lora/a", "binary", "org/model")
    _lora(tmp_path, "lora/b", "ternary", "org/model")

    models = discover_models(tmp_path)

    assert [(model.family, model.task) for model in models].count(("bert", "binary")) == 1
    assert [(model.family, model.task) for model in models].count(("bert", "ternary")) == 1
    assert len([model for model in models if model.family == "lora"]) == 2
    bases = [model for model in models if model.family == "base_llm"]
    assert {model.task for model in bases} == {"binary", "ternary"}
    assert len(bases) == 2


def test_json_accepts_separate_bert_paths_and_base_tasks(tmp_path: Path) -> None:
    manifest = tmp_path / "models.json"
    _write_json(
        manifest,
        {
            "models": [
                {"name": "bert-b", "family": "bert", "task": "binary", "path": "b"},
                {"name": "bert-t", "family": "bert", "task": "ternary", "path": "t"},
                {"name": "base", "family": "base_llm", "model_id": "org/model"},
            ]
        },
    )

    models = resolve_models_source(manifest)

    assert {model.name for model in models} == {"bert-b", "bert-t", "base__binary", "base__ternary"}
