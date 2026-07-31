from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from openpyxl import load_workbook

from jura_hypersumm.reporting import write_document_review_package


def test_document_review_package_contains_model_and_top1_rag_workbooks(
    tmp_path: Path,
) -> None:
    pairs = pd.DataFrame(
        [
            {
                "document": "decision.docx",
                "task": "binary",
                "hypothesis_id": "decision.docx:00000",
                "sentence_index": 0,
                "hypothesis": "Sentence one.",
                "premise": "second-ranked text",
                "source": "КоАП РФ: Статья 18.8. п. 2.",
                "retrieval_rank": 2,
                "prediction": "no",
                "citation_article": "18.8",
            },
            {
                "document": "decision.docx",
                "task": "binary",
                "hypothesis_id": "decision.docx:00000",
                "sentence_index": 0,
                "hypothesis": "Sentence one.",
                "premise": "top-ranked text",
                "source": "КоАП РФ: Статья 18.8. п. 1.",
                "retrieval_rank": 1,
                "prediction": "contradiction",
                "citation_article": "18.8",
            },
            {
                "document": "decision.docx",
                "task": "ternary",
                "hypothesis_id": "decision.docx:00000",
                "sentence_index": 0,
                "hypothesis": "Sentence one.",
                "premise": "top-ranked text",
                "source": "КоАП РФ: Статья 18.8. п. 1.",
                "retrieval_rank": 1,
                "prediction": "entailment",
                "citation_article": "18.8",
            },
            {
                "document": "decision.docx",
                "task": "binary",
                "hypothesis_id": "decision.docx:00001",
                "sentence_index": 1,
                "hypothesis": "Sentence two.",
                "premise": "article text 30.3",
                "source": "КоАП РФ: Статья 30.3. п. 1.",
                "retrieval_rank": 1,
                "prediction": "no",
                "citation_article": "30.3",
            },
        ]
    )

    archive_path = write_document_review_package(
        "ready_llm", pairs, output_dir=tmp_path
    )

    assert archive_path is not None
    with ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "decision_binary_model_predictions.xlsx",
            "decision_ternary_model_predictions.xlsx",
            "decision_rag_retrieval.xlsx",
        }
        model = pd.read_excel(
            BytesIO(archive.read("decision_binary_model_predictions.xlsx"))
        )
        assert list(model.columns) == [
            "hypothesis",
            "premise",
            "model_prediction",
            "expert_label",
            "expert_comment",
        ]
        assert len(model) == 3

        rag = pd.read_excel(BytesIO(archive.read("decision_rag_retrieval.xlsx")))
        assert list(rag.columns) == [
            "sentence",
            "article_number",
            "article_text",
        ]
        assert rag.to_dict(orient="records") == [
            {
                "sentence": "Sentence one.",
                "article_number": 18.8,
                "article_text": "top-ranked text",
            },
            {
                "sentence": "Sentence two.",
                "article_number": 30.3,
                "article_text": "article text 30.3",
            },
        ]
        rag_workbook = load_workbook(
            BytesIO(archive.read("decision_rag_retrieval.xlsx")), read_only=True
        )
        article_cell = rag_workbook["rag_retrieval"]["B2"]
        assert article_cell.value == "18.8"
        assert article_cell.data_type == "s"


def test_document_review_package_is_skipped_without_pairs(tmp_path: Path) -> None:
    assert (
        write_document_review_package(
            "lora", pd.DataFrame(), output_dir=tmp_path
        )
        is None
    )
