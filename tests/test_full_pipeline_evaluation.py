import json
from pathlib import Path

import pandas as pd

from jura_hypersumm.full_pipeline import run_full_pipeline_evaluation
from jura_hypersumm.rag.artifacts import RagBundle


def test_full_pipeline_state_resumes_completed_jobs(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "bert-binary",
                        "family": "bert",
                        "task": "binary",
                        "path": "artifact",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "prompt.py").write_text('PROMPT_TEXT = "t"', encoding="utf-8")
    (tmp_path / "prompt_binary.py").write_text(
        'PROMPT_TEXT_BIN = "b"', encoding="utf-8"
    )
    bundle = RagBundle(
        "test-rag", tmp_path / "codex.csv", tmp_path / "index", "encoder", None, False
    )
    monkeypatch.setattr("jura_hypersumm.full_pipeline.load_rag_bundle", lambda path: bundle)
    monkeypatch.setattr(
        "jura_hypersumm.full_pipeline.PremiseRetriever.from_components",
        lambda *args, **kwargs: object(),
    )
    calls = []
    monkeypatch.setattr(
        "jura_hypersumm.full_pipeline._load_predictor",
        lambda entry, prompt, parameters: (object(), object(), object()),
    )

    def fake_evaluate(entry, **kwargs):
        calls.append(entry.name)
        return {
            "scores": pd.DataFrame(
                [{"model": entry.name, "task": entry.task, "evaluation_scope": "validation"}]
            ),
            "details": pd.DataFrame([{"value": 1}]),
        }

    monkeypatch.setattr("jura_hypersumm.full_pipeline._evaluate_one", fake_evaluate)
    results = tmp_path / "results"

    first = run_full_pipeline_evaluation(
        models_source=manifest,
        rag_source=tmp_path / "rag.json",
        repo_root=tmp_path,
        results_dir=results,
    )
    second = run_full_pipeline_evaluation(
        models_source=manifest,
        rag_source=tmp_path / "rag.json",
        repo_root=tmp_path,
        results_dir=results,
    )

    assert calls == ["bert-binary"]
    assert first["model_name"].tolist() == ["bert-binary"]
    assert second["model_name"].tolist() == ["bert-binary"]


def test_legacy_retrieval_top_k_override_sets_both_depths(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "bert-binary",
                        "family": "bert",
                        "task": "binary",
                        "path": "artifact",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "prompt.py").write_text('PROMPT_TEXT = "t"', encoding="utf-8")
    (tmp_path / "prompt_binary.py").write_text(
        'PROMPT_TEXT_BIN = "b"', encoding="utf-8"
    )
    bundle = RagBundle(
        "test-rag", tmp_path / "codex.csv", tmp_path / "index", "encoder", None, False
    )
    monkeypatch.setattr("jura_hypersumm.full_pipeline.load_rag_bundle", lambda path: bundle)
    monkeypatch.setattr(
        "jura_hypersumm.full_pipeline.PremiseRetriever.from_components",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "jura_hypersumm.full_pipeline._load_predictor",
        lambda entry, prompt, parameters: (object(), object(), object()),
    )
    observed = {}

    def fake_evaluate(entry, **kwargs):
        observed.update(kwargs["parameters"])
        return {"scores": pd.DataFrame([{"model": entry.name}])}

    monkeypatch.setattr("jura_hypersumm.full_pipeline._evaluate_one", fake_evaluate)

    run_full_pipeline_evaluation(
        models_source=manifest,
        rag_source=tmp_path / "rag.json",
        repo_root=tmp_path,
        results_dir=tmp_path / "results",
        inference_parameters={"retrieval_top_k": 37},
    )

    assert observed["candidate_top_k"] == 37
    assert observed["final_top_k"] == 37
