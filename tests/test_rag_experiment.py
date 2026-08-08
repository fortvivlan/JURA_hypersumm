from pathlib import Path

import pandas as pd
import pytest

from jura_hypersumm.rag.data import convert_embedding_dataset
from jura_hypersumm.rag.evaluation import (
    _comparison_deltas,
    calculate_recall_rows,
    read_rag_workbook,
)
from jura_hypersumm.rag.artifacts import RagBundle
from jura_hypersumm.retrieval import Citation


def test_embedding_conversion_merges_only_requested_labels(tmp_path: Path) -> None:
    source = tmp_path / "train.xlsx"
    pd.DataFrame(
        {
            "premise": ["p1", "p2", "p3"],
            "hypothesis": ["h1", "h2", "h3"],
            "source": ["s1", "s2", "s3"],
            "tag": ["contradiction", "entailment", "not mentioned"],
        }
    ).to_excel(source, index=False)

    converted = convert_embedding_dataset(source, "train")

    assert converted["embedding_tag"].tolist() == ["similar", "similar", "not mentioned"]
    assert converted["label"].tolist() == [1, 1, 0]
    assert converted["example_id"].tolist() == ["train:000000", "train:000001", "train:000002"]


def test_headerless_rag_workbook_groups_multiple_and_zero_golds(tmp_path: Path) -> None:
    workbook = tmp_path / "rag.xlsx"
    pd.DataFrame(
        [
            ["Sentence one", "КоАП РФ Статья 18.8 Часть 3.1", "text 1"],
            ["Sentence one", "КоАП РФ Статья 3.10 Часть 5", "text 2"],
            ["Sentence two", "", ""],
        ]
    ).to_excel(workbook, header=False, index=False)

    rows = read_rag_workbook(workbook)

    assert len(rows) == 2
    assert len(rows[0]["gold_citations"]) == 2
    assert rows[0]["gold_citations"][0].part == "3.1"
    assert rows[1]["gold_citations"] == ()


def test_recall_curves_report_query_and_article_micro_recall() -> None:
    one = ("коап рф", "1", "1", "")
    two = ("коап рф", "2", "", "")
    rows = [
        {"gold_keys": {one, two}, "retrieved_keys": [one, two], "method": "faiss"},
        {"gold_keys": {one}, "retrieved_keys": [two, one], "method": "exact"},
    ]

    scores = calculate_recall_rows(rows, rag_name="rag", workbook_name="test")
    at_one = next(row for row in scores if row["branch"] == "all" and row["cutoff"] == 1)
    at_five = next(row for row in scores if row["branch"] == "all" and row["cutoff"] == 5)

    assert at_one["query_recall"] == pytest.approx(0.5)
    assert at_one["article_micro_recall"] == pytest.approx(1 / 3)
    assert at_five["query_recall"] == 1
    assert at_five["article_micro_recall"] == 1


def test_candidate_recall_and_dynamic_final_cutoffs() -> None:
    gold = ("коап рф", "1", "", "")
    rows = [
        {
            "gold_keys": {gold},
            "candidate_keys": [("коап рф", "2", "", ""), gold],
            "retrieved_keys": [("коап рф", "2", "", "")],
            "method": "faiss",
        }
    ]

    scores = calculate_recall_rows(
        rows,
        rag_name="rag",
        workbook_name="test",
        cutoffs=(1,),
        candidate_top_k=2,
    )

    candidate = next(row for row in scores if row["branch"] == "all" and row["stage"] == "candidate")
    final = next(row for row in scores if row["branch"] == "all" and row["stage"] == "final")
    assert candidate["query_recall"] == 1
    assert final["query_recall"] == 0


def test_four_way_comparison_deltas() -> None:
    variants = {
        ("base", False): "base__no",
        ("base", True): "base__yes",
        ("tuned", False): "tuned__no",
        ("tuned", True): "tuned__yes",
    }
    scores = pd.DataFrame(
        [
            {
                "rag_version": name,
                "workbook": "w",
                "branch": "all",
                "stage": "final",
                "cutoff": 1,
                "query_recall": value,
                "article_micro_recall": value,
            }
            for name, value in (
                ("base__no", 0.2),
                ("base__yes", 0.3),
                ("tuned__no", 0.4),
                ("tuned__yes", 0.6),
            )
        ]
    )
    bundles = [
        RagBundle("base", Path("c"), Path("i"), "e", None, False),
        RagBundle("tuned", Path("c"), Path("i"), "e", None, False),
    ]

    deltas = _comparison_deltas(scores, bundles, variants, True)

    values = deltas.set_index("comparison")["query_recall"].to_dict()
    assert values["embedding_without_reranker"] == pytest.approx(0.2)
    assert values["reranker_on_baseline"] == pytest.approx(0.1)
    assert values["reranker_on_tuned"] == pytest.approx(0.2)
    assert values["combined_vs_unchanged_baseline"] == pytest.approx(0.4)
