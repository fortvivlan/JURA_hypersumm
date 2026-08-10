import json
from pathlib import Path

import pandas as pd

from jura_hypersumm.common import merge_parameters, resolve_model
from jura_hypersumm.lora import DEFAULT_LORA_HYPERPARAMETERS
from jura_hypersumm import lora_sweep as sweep


def _state(models=("qwen", "llama", "ministral", "t-lite"), tasks=("binary", "ternary")):
    base = merge_parameters(
        DEFAULT_LORA_HYPERPARAMETERS, sweep.HISTORICAL_LORA_OVERRIDES
    )
    return {
        "configuration": {
            "models": list(models),
            "tasks": list(tasks),
            "base_hyperparameters": base,
            "dataset_sha256": {
                f"{split}_{task}": f"{split}-{task}"
                for split in ("train", "val")
                for task in tasks
            },
            "prompt_sha256": {task: f"prompt-{task}" for task in tasks},
        },
        "pins": {
            "model_revisions": {model: model * 8 for model in models},
            "rag_revision": "b" * 40,
        },
        "stages": {},
        "experiments": {},
    }


def _complete_with_label(state, stage, label):
    candidates = sweep.build_stage_candidates(stage, state)
    stage_state = sweep._register_stage(state, stage, candidates)
    winners = {}
    for candidate in stage_state["candidates"].values():
        if candidate["label"] == label:
            winners[f"{candidate['model_alias']}:{candidate['task']}"] = candidate
    stage_state["winners"] = winners
    stage_state["status"] = "completed"
    return candidates


def test_coordinate_search_has_120_logical_and_88_unique_recipes() -> None:
    state = _state()
    logical_counts = []
    new_counts = []
    labels = {
        "target_modules": "qv",
        "rank": "r_16",
        "learning_rate": "lr_2e-5",
        "alpha": "alpha_2r",
        "dropout": "dropout_0.1",
    }
    previous_total = 0

    for stage in sweep.STAGE_ORDER:
        candidates = _complete_with_label(state, stage, labels[stage])
        logical_counts.append(len(candidates))
        new_counts.append(len(state["experiments"]) - previous_total)
        previous_total = len(state["experiments"])

    assert logical_counts == [24, 24, 32, 16, 24]
    assert new_counts == [24, 16, 24, 8, 16]
    assert sum(logical_counts) == 120
    assert len(state["experiments"]) == 88


def test_stage_grids_and_inheritance_are_exact() -> None:
    state = _state(models=("qwen",), tasks=("binary",))
    targets = _complete_with_label(state, "target_modules", "qkv")
    assert [candidate.parameters["target_modules"] for candidate in targets] == [
        ["q_proj", "v_proj"],
        ["q_proj", "k_proj", "v_proj"],
        "all-linear",
    ]

    ranks = _complete_with_label(state, "rank", "r_32")
    assert [candidate.parameters["lora_rank"] for candidate in ranks] == [8, 16, 32]
    assert [candidate.parameters["lora_alpha"] for candidate in ranks] == [16, 32, 64]
    assert all(
        candidate.parameters["target_modules"] == ["q_proj", "k_proj", "v_proj"]
        for candidate in ranks
    )

    rates = _complete_with_label(state, "learning_rate", "lr_1e-4")
    assert [candidate.parameters["learning_rate"] for candidate in rates] == [
        2e-5,
        1e-4,
        2e-4,
        1e-5,
    ]

    alphas = _complete_with_label(state, "alpha", "alpha_1r")
    assert [candidate.parameters["lora_alpha"] for candidate in alphas] == [32, 64]

    dropouts = sweep.build_stage_candidates("dropout", state)
    assert [candidate.parameters["lora_dropout"] for candidate in dropouts] == [
        0.0,
        0.05,
        0.1,
    ]


def test_winner_uses_validation_not_benchmark() -> None:
    state = _state(models=("qwen",), tasks=("binary",))
    candidates = sweep.build_stage_candidates("target_modules", state)
    sweep._register_stage(state, "target_modules", candidates)
    rows = []
    for index, candidate in enumerate(candidates):
        recipe_id = sweep._recipe_id(
            candidate.model_alias, candidate.task, candidate.parameters, state
        )
        rows.extend(
            [
                {
                    "recipe_id": recipe_id,
                    "evaluation_scope": "validation",
                    "macro_f1": 0.9 - index * 0.1,
                    "contradiction_f1": 0.8,
                    "invalid_predictions": 0,
                },
                {
                    "recipe_id": recipe_id,
                    "evaluation_scope": "autotest_total",
                    "macro_f1": 0.1 + index * 0.4,
                    "contradiction_f1": 0.8,
                    "invalid_predictions": 0,
                },
            ]
        )

    sweep._rank_and_finalize(state, "target_modules", pd.DataFrame(rows))

    assert state["stages"]["target_modules"]["winners"]["qwen:binary"]["label"] == "qv"


def test_adapter_reuse_requires_complete_matching_manifest(tmp_path: Path) -> None:
    state = _state(models=("qwen",), tasks=("binary",))
    candidate = sweep.build_stage_candidates("target_modules", state)[0]
    sweep._register_stage(state, "target_modules", [candidate])
    experiment = next(iter(state["experiments"].values()))
    target = sweep._adapter_target(tmp_path, experiment)
    target.mkdir(parents=True)
    for name in ("adapter_config.json", "tokenizer_config.json"):
        (target / name).write_text("{}", encoding="utf-8")
    (target / "adapter_model.safetensors").write_bytes(b"adapter")
    manifest = {
        "model_id": resolve_model("qwen").model_id,
        "task": "binary",
        "resolved_revision": state["pins"]["model_revisions"]["qwen"],
        "train_sha256": "train-binary",
        "validation_sha256": "val-binary",
        "prompt_sha256": "prompt-binary",
        "prompt_processing": "standard_chat_template_v1",
        "premise_format": "source_prefixed_v1",
        "rag_revision": "b" * 40,
        "hyperparameters": dict(experiment["parameters"]),
    }
    (target / "run_config.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert sweep._adapter_matches(tmp_path, experiment, state)
    manifest["hyperparameters"]["learning_rate"] = 1e-6
    (target / "run_config.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert not sweep._adapter_matches(tmp_path, experiment, state)


def test_latest_checkpoint_uses_highest_numeric_step(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-9").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-final").mkdir()

    assert sweep._latest_checkpoint(tmp_path) == tmp_path / "checkpoint-100"


def test_empty_scores_do_not_complete_a_stage() -> None:
    state = _state(models=("qwen",), tasks=("binary",))
    candidates = sweep.build_stage_candidates("target_modules", state)
    sweep._register_stage(state, "target_modules", candidates)

    assert not sweep._stage_complete(state, "target_modules", pd.DataFrame())


def test_completed_stage_writes_all_result_sheets(tmp_path: Path) -> None:
    state = _state(models=("qwen",), tasks=("binary",))
    candidates = sweep.build_stage_candidates("target_modules", state)
    sweep._register_stage(state, "target_modules", candidates)
    rows = []
    for index, candidate in enumerate(candidates):
        recipe_id = sweep._recipe_id(
            candidate.model_alias, candidate.task, candidate.parameters, state
        )
        state["experiments"][recipe_id]["status"] = "completed"
        for dataset, scope in (
            (None, "validation"),
            ("Dialogue", "autotest_model"),
            ("Dialogue", "autotest_total"),
            ("Full", "autotest_model"),
            ("Full", "autotest_total"),
        ):
            rows.append(
                {
                    "recipe_id": recipe_id,
                    "model_alias": "qwen",
                    "task": "binary",
                    "evaluation_scope": scope,
                    "test_dataset": dataset,
                    "macro_f1": 0.8 - index * 0.1,
                    "contradiction_f1": 0.7,
                    "invalid_predictions": 0,
                }
            )
    scores = pd.DataFrame(rows)
    sweep._rank_and_finalize(state, "target_modules", scores)

    sweep._write_stage_artifacts(state, "target_modules", scores, tmp_path)

    stage_dir = tmp_path / "stages/target_modules"
    assert (stage_dir / "scores.csv").is_file()
    workbook = pd.ExcelFile(stage_dir / "results.xlsx")
    assert workbook.sheet_names == [
        "scores",
        "validation_ranking",
        "benchmark_ranking",
        "winners",
        "candidates",
        "experiments",
        "failures",
    ]


def test_invocation_attempt_cap_is_enforced(monkeypatch, tmp_path: Path) -> None:
    base_configuration = {}

    def fake_configuration(root, models, tasks, base):
        del root
        configuration = {
            "models": list(models),
            "tasks": list(tasks),
            "base_hyperparameters": dict(base),
            "dataset_sha256": {
                f"{split}_{task}": f"{split}-{task}"
                for split in ("train", "val")
                for task in tasks
            },
            "prompt_sha256": {task: f"prompt-{task}" for task in tasks},
        }
        base_configuration.update(configuration)
        return configuration

    monkeypatch.setattr(sweep, "_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(sweep, "_configuration", fake_configuration)
    monkeypatch.setattr(
        sweep,
        "_resolve_pins",
        lambda root, models: {
            "model_revisions": {model: "a" * 40 for model in models},
            "rag_revision": "b" * 40,
        },
    )
    monkeypatch.setattr(
        sweep, "_run_experiment", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(sweep, "_write_stage_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(sweep, "_cleanup_cuda", lambda logger: None)

    sweep.run_sweep_stage(
        "target_modules",
        search_id="attempt-cap",
        repo_root=tmp_path,
        models=("qwen",),
        tasks=("binary",),
        max_attempts_per_run=2,
        max_retries=0,
    )

    state_path = tmp_path / "local_results/lora_searches/attempt-cap/search_state.json"
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert sum(record["attempts"] for record in saved["experiments"].values()) == 2
    assert sum(record["status"] == "failed" for record in saved["experiments"].values()) == 2
    assert sum(record["status"] == "pending" for record in saved["experiments"].values()) == 1
