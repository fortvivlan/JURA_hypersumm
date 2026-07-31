from pathlib import Path

import pandas as pd
import pytest

from jura_hypersumm.common import (
    evaluate_predictions,
    load_dataset,
    merge_parameters,
    resolve_model,
)


def test_resolve_model_accepts_alias_and_full_id() -> None:
    assert resolve_model("qwen").model_id == "Qwen/Qwen3-8B"
    assert resolve_model("Qwen/Qwen3-8B").alias == "qwen"


def test_resolve_model_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        resolve_model("unknown/model")


def test_merge_parameters_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        merge_parameters({"epochs": 3}, {"epoch": 2})


def test_load_dataset_uses_explicit_task_labels(tmp_path: Path) -> None:
    path = tmp_path / "binary.csv"
    pd.DataFrame(
        {
            "premise": ["p"],
            "hypothesis": ["h"],
            "source": ["s"],
            "tag": ["entailment"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid binary labels"):
        load_dataset(path, "binary")


def test_evaluate_predictions_counts_invalid_as_wrong() -> None:
    dataframe = pd.DataFrame(
        {
            "example_id": ["v:0", "v:1", "v:2"],
            "premise": ["p0", "p1", "p2"],
            "hypothesis": ["h0", "h1", "h2"],
            "source": ["s0", "s1", "s2"],
            "tag": ["contradiction", "no", "no"],
        }
    )

    tables = evaluate_predictions(
        dataframe,
        ["contradiction", None, "no"],
        ["contradiction", "unparseable", "no"],
        model_id="model",
        task="binary",
    )

    score = tables.scores.iloc[0]
    assert score["accuracy"] == pytest.approx(2 / 3)
    assert score["invalid_predictions"] == 1
    assert set(tables.per_class["label"]) == {"contradiction", "no"}
    invalid_column = tables.confusion_matrix[
        tables.confusion_matrix["predicted_label"] == "invalid"
    ]
    assert invalid_column["count"].sum() == 1
