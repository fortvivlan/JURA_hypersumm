from dataclasses import dataclass
from pathlib import Path
import subprocess

import pandas as pd

from jura_hypersumm.retrieval import (
    PremiseRetriever,
    ensure_rag_repository,
    extract_citation,
)


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict[str, str]


class FakeVectorStore:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def similarity_search_with_score(self, query: str, k: int):
        self.calls.append((query, k))
        return [
            (FakeDocument(f"semantic {index}", {"source": f"s{index}"}), index / 10)
            for index in range(k)
        ]


def _retriever() -> tuple[PremiseRetriever, FakeVectorStore]:
    codex = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "source": [
                "КоАП РФ: Статья 5.35. п. 1. ",
                "КоАП РФ: Статья 5.35.1. п. 1. ",
                "КоАП РФ: Статья 18.8. п. 3. ",
                "КоАП РФ: Статья 18.8. п. 3.1.",
            ],
        }
    )
    vectorstore = FakeVectorStore()
    return PremiseRetriever(codex, vectorstore), vectorstore


def test_extract_citation_preserves_dotted_numbers_and_long_code_name() -> None:
    citation = extract_citation(
        "по ч. 3.1 ст. 18.8 Кодекса Российской Федерации "
        "об административных правонарушениях"
    )
    assert citation.article == "18.8"
    assert citation.part == "3.1"
    assert citation.code == "КоАП РФ"


def test_exact_lookup_does_not_confuse_dotted_article_prefixes() -> None:
    retriever, vectorstore = _retriever()

    records = retriever.retrieve("п. 1 ст. 5.35 КоАП РФ")

    assert [record.source for record in records] == [
        "КоАП РФ: Статья 5.35. п. 1. "
    ]
    assert records[0].method == "exact"
    assert vectorstore.calls == []


def test_exact_lookup_distinguishes_point_3_from_3_1() -> None:
    retriever, _ = _retriever()
    records = retriever.retrieve("ч. 3.1 ст. 18.8 КоАП РФ")
    assert len(records) == 1
    assert "п. 3.1." in records[0].source


def test_ambiguous_citation_falls_back_to_at_most_twenty() -> None:
    retriever, vectorstore = _retriever()

    records = retriever.retrieve("ст. 18.8 КоАП РФ", top_k=200)

    assert len(records) == 20
    assert all(record.method == "faiss" for record in records)
    assert vectorstore.calls == [("ст. 18.8 КоАП РФ", 20)]


def test_rag_repository_is_checked_out_at_requested_commit(tmp_path: Path) -> None:
    repository = tmp_path / "rag"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"],
        check=True,
    )
    (repository / "faiss_index").mkdir()
    (repository / "faiss_index" / "index.faiss").write_bytes(b"index")
    (repository / "faiss_index" / "index.pkl").write_bytes(b"metadata")
    (repository / "codex.csv").write_text("text,source\na,s\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "artifacts"],
        check=True,
    )
    pinned = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (repository / "README.md").write_text("later change", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "later"],
        check=True,
    )

    path, resolved = ensure_rag_repository(repository, revision=pinned)

    assert path == repository
    assert resolved == pinned
