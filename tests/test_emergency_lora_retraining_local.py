import importlib.util
import json
import sys
from pathlib import Path

from jura_hypersumm.common import resolve_model, slugify_model_id


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "run_emergency_lora_retraining_local.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_emergency_lora_retraining_local", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_dry_run_reuses_existing_campaign_recipe(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (tmp_path / "jura_hypersumm").mkdir()
    spec = resolve_model("ministral")
    target = (
        tmp_path
        / "local_artifacts"
        / "campaigns"
        / "full_pipeline_v1"
        / "models"
        / "lora"
        / slugify_model_id(spec.model_id)
        / "ternary"
    )
    target.mkdir(parents=True)
    (target / "run_config.json").write_text(
        json.dumps(
            {
                "model_id": spec.model_id,
                "task": "ternary",
                "resolved_revision": "a" * 40,
                "hyperparameters": {"epochs": 3},
            }
        ),
        encoding="utf-8",
    )
    (target / "adapter_config.json").write_text("{}", encoding="utf-8")
    (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (target / "adapter_model.safetensors").write_bytes(b"adapter")

    plan = MODULE.run_emergency_lora_retraining(
        repo_root=tmp_path,
        models=("ministral",),
        tasks=("ternary",),
        dry_run=True,
    )

    assert plan[["model_alias", "task", "status"]].to_dict("records") == [
        {"model_alias": "ministral", "task": "ternary", "status": "planned"}
    ]
    assert plan.iloc[0]["target"] == str(target)
