from pathlib import Path

import pandas as pd
import pytest

from convert_datasets import _legacy_article_prefix, _to_binary_label, run_conversion


def _write_dataset(path: Path, tags: list[str]) -> None:
    dataframe = pd.DataFrame(
        {
            "premise": [f"premise {index}" for index in range(len(tags))],
            "hypothesis": [f"hypothesis {index}" for index in range(len(tags))],
            "source": [f"source {index}" for index in range(len(tags))],
            "tag": tags,
        }
    )
    dataframe.to_excel(path, index=False, engine="openpyxl")


@pytest.mark.parametrize(
    ("ternary_label", "binary_label"),
    [
        ("contradiction", "contradiction"),
        ("entailment", "no"),
        ("not mentioned", "no"),
    ],
)
def test_to_binary_label(ternary_label: str, binary_label: str) -> None:
    assert _to_binary_label(ternary_label) == binary_label


def test_to_binary_label_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="Unexpected label"):
        _to_binary_label("unknown")


@pytest.mark.parametrize(
    ("source", "prefix"),
    [
        ("КоАП Статья 18.8 Часть 3.1", "ч.3.1 ст.18.8 КоАП РФ"),
        (
            "КоАП Статья 27.19.1 Часть 1",
            "КоАП Статья 27.19.1 Часть 1",
        ),
        ("КоАП РФ Статья 20.20 Часть 3", "ч.3 ст.20.20 РФ РФ"),
        (
            "КоАП Статья Статья 18.15 Часть 4",
            "ч.4 ст.18.15 Статья РФ",
        ),
        ("", ""),
    ],
)
def test_legacy_article_prefix_reproduces_original_format(
    source: str, prefix: str
) -> None:
    assert _legacy_article_prefix(source) == prefix


def test_run_conversion_writes_both_task_types_for_both_splits(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.xlsx"
    val_path = tmp_path / "val.xlsx"
    output_dir = tmp_path / "csv"
    _write_dataset(
        train_path, ["contradiction", "entailment", "not mentioned"]
    )
    _write_dataset(val_path, ["not mentioned", "contradiction"])

    output_paths = run_conversion(train_path, val_path, output_dir)

    assert set(output_paths) == {
        "train_ternary",
        "train_binary",
        "val_ternary",
        "val_binary",
    }
    assert pd.read_csv(output_paths["train_ternary"])["tag"].tolist() == [
        "contradiction",
        "entailment",
        "not mentioned",
    ]
    assert pd.read_csv(output_paths["train_binary"])["tag"].tolist() == [
        "contradiction",
        "no",
        "no",
    ]
    assert pd.read_csv(output_paths["val_binary"])["tag"].tolist() == [
        "no",
        "contradiction",
    ]
    ternary = pd.read_csv(output_paths["train_ternary"])
    assert ternary.columns.tolist() == ["premise", "hypothesis", "tag"]
    assert ternary["premise"].tolist() == [
        "source 0 premise 0",
        "source 1 premise 1",
        "source 2 premise 2",
    ]


def test_run_conversion_rejects_invalid_tags_before_writing(tmp_path: Path) -> None:
    train_path = tmp_path / "train.xlsx"
    val_path = tmp_path / "val.xlsx"
    output_dir = tmp_path / "csv"
    _write_dataset(train_path, ["contradiction"])
    _write_dataset(val_path, ["invalid"])

    with pytest.raises(ValueError, match="unexpected labels"):
        run_conversion(train_path, val_path, output_dir)

    assert not output_dir.exists()
