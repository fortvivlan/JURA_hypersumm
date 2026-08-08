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
        detected = (Citation("КоАП РФ", "32.9", "1", None),)
        return [
            RetrievalRecord(
                "p1", "source 1", "exact", 1, None, detected[0], detected
            ),
            RetrievalRecord(
                "p2", "source 2", "exact", 2, None, detected[0], detected
            ),
        ]


class TrackingRetriever(FakeRetriever):
    def __init__(self) -> None:
        self.hypotheses = []

    def retrieve(self, hypothesis: str, *, top_k: int):
        self.hypotheses.append(hypothesis)
        return super().retrieve(hypothesis, top_k=top_k)


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
    assert '"article": "32.9"' in contradiction["detected_citations"]
    assert tables.aggregates.iloc[0]["unresolved_citations"] == "[]"


def test_document_inference_filters_irrelevant_sentences_before_retrieval(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "jura_hypersumm.inference.read_docx_text", lambda path: "ПОСТАНОВИЛ: text"
    )
    monkeypatch.setattr(
        "jura_hypersumm.inference.split_russian_sentences",
        lambda text: [
            "Судья Петрова.",
            "Оплатить по реквизитам.",
            "Банковские ре...изиты.",
            "Приложить квитанцию.",
            "Банк получателя, БИК 044525000.",
            "Получатель, ИНН 7700000000.",
            "Номер начисления, УИН 18810177240010001111.",
            "Код дохода, КБК 18811601181019000140.",
            "Код территории, ОКТМО 45382000.",
            "Лицевой счет, л/с 40100770005.",
            "Расчетный счет, р/с 40101810045250010041.",
            "40101810045250010041.",
            "Назначить административный штраф.",
        ],
    )
    document_path = tmp_path / "decision.docx"
    document_path.write_bytes(b"stable document")
    retriever = TrackingRetriever()

    tables = run_document_inference(
        [document_path],
        predictor=FakePredictor(),
        retriever=retriever,
        model_id="model",
        task="binary",
    )

    assert retriever.hypotheses == ["Назначить административный штраф."]
    assert set(tables.pairs["hypothesis"]) == {"Назначить административный штраф."}
    assert set(tables.pairs["sentence_index"]) == {12}
    assert set(tables.pairs["hypothesis_id"]) == {"decision.docx:00012"}


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
