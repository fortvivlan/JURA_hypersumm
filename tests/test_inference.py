from pathlib import Path

from jura_hypersumm.inference import (
    ModelPrediction,
    aggregate_pair_labels,
    run_document_inference,
)
from jura_hypersumm.retrieval import Citation, RetrievalRecord


class FakeRetriever:
    def retrieve(self, hypothesis: str, *, top_k: int):
        assert top_k <= 20
        return [
            RetrievalRecord("p1", "source 1", "exact", 1, None, Citation()),
            RetrievalRecord("p2", "source 2", "exact", 2, None, Citation()),
        ]


class FakePredictor:
    def predict_pairs(self, premises, hypothesis):
        return [
            ModelPrediction("no", "no"),
            ModelPrediction("contradiction", "contradiction"),
        ]


def test_aggregate_pair_labels() -> None:
    assert aggregate_pair_labels(["entailment", "contradiction"], "ternary") == "contradiction"
    assert aggregate_pair_labels([None], "binary") == "invalid"
    assert aggregate_pair_labels(["not mentioned", "entailment"], "ternary") == "entailment"


def test_document_inference_preserves_contradiction_premise(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "jura_hypersumm.inference.read_docx_text", lambda path: "ПОСТАНОВИЛ: sentence"
    )
    monkeypatch.setattr(
        "jura_hypersumm.inference.split_russian_sentences", lambda text: ["sentence"]
    )

    document_path = tmp_path / "decision.docx"
    document_path.write_bytes(b"stable document")
    tables = run_document_inference(
        [document_path],
        predictor=FakePredictor(),
        retriever=FakeRetriever(),
        model_id="bert",
        task="binary",
    )

    assert tables.aggregates.iloc[0]["prediction"] == "contradiction"
    contradiction = tables.pairs[tables.pairs["prediction"] == "contradiction"].iloc[0]
    assert contradiction["premise"] == "p2"
    assert contradiction["source"] == "source 2"
    assert len(contradiction["document_sha256"]) == 64


def test_missing_operative_section_is_reported_and_skipped(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "jura_hypersumm.inference.read_docx_text", lambda path: "No marker"
    )
    document_path = tmp_path / "decision.docx"
    document_path.write_bytes(b"stable document")
    tables = run_document_inference(
        [document_path],
        predictor=FakePredictor(),
        retriever=FakeRetriever(),
        model_id="model",
        task="binary",
    )
    assert tables.pairs.empty
    assert "not found" in tables.errors.iloc[0]["error"]
