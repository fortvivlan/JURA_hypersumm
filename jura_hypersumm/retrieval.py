"""Deterministic legal-citation lookup followed by bounded FAISS retrieval."""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_RAG_REPOSITORY,
    DEFAULT_RAG_REVISION,
)


@dataclass(frozen=True)
class Citation:
    """Structured Russian legal citation fields, kept as strings."""

    code: str | None = None
    article: str | None = None
    part: str | None = None
    point: str | None = None


@dataclass(frozen=True)
class RetrievalRecord:
    """One premise and its retrieval audit metadata."""

    premise: str
    source: str
    method: str
    rank: int
    score: float | None
    citation: Citation


_CODE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("КоАП РФ", r"(?:КоАП(?:\s+РФ)?|Кодекс[а-я\s]+об\s+административных\s+правонарушениях)"),
    ("УПК РФ", r"(?:УПК(?:\s+РФ)?|Уголовно-процессуальн[а-я\s]+кодекс[а-я\s]*(?:РФ)?)"),
    ("УК РФ", r"(?:УК(?:\s+РФ)?|Уголовн[а-я\s]+кодекс[а-я\s]*(?:РФ)?)"),
    ("ГПК РФ", r"(?:ГПК(?:\s+РФ)?|Гражданск[а-я\s-]+процессуальн[а-я\s]+кодекс[а-я\s]*(?:РФ)?)"),
    ("ГК РФ", r"(?:ГК(?:\s+РФ)?|Гражданск[а-я\s]+кодекс[а-я\s]*(?:РФ)?)"),
    ("КАС РФ", r"(?:КАС(?:\s+РФ)?|Кодекс[а-я\s]+административного\s+судопроизводства)"),
    ("УИК РФ", r"(?:УИК(?:\s+РФ)?|Уголовно-исполнительн[а-я\s]+кодекс[а-я\s]*(?:РФ)?)"),
    ("ТК", r"(?:ТК(?:\s+РФ)?|Трудов[а-я\s]+кодекс[а-я\s]*(?:РФ)?)"),
)
_ARTICLE_PATTERN = re.compile(
    r"(?i)(?:\bст(?:атья|атьи|атью|атье|атьей|атьёй)?\.?)[\s№]*([0-9]+(?:\.[0-9]+)*)"
)
_SUBPROVISION_PATTERN = re.compile(
    r"(?i)\b(ч(?:асть|асти|астью)?|п(?:ункт|ункта|унктом)?|ч|п)\.?\s*([0-9]+(?:[.-][0-9]+)*)"
)


def extract_citation(text: str) -> Citation:
    """Extract code, article, part, and point without numeric conversion."""
    code = next(
        (name for name, pattern in _CODE_PATTERNS if re.search(pattern, text, re.I)),
        None,
    )
    article_match = _ARTICLE_PATTERN.search(text)
    article = article_match.group(1) if article_match else None
    part = point = None
    if article_match:
        window_start = max(0, article_match.start() - 100)
        window_end = min(len(text), article_match.end() + 100)
        for kind, number in _SUBPROVISION_PATTERN.findall(text[window_start:window_end]):
            if kind.lower().startswith("ч"):
                part = number
            else:
                point = number
    return Citation(code=code, article=article, part=part, point=point)


def ensure_rag_repository(
    rag_dir: str | Path,
    repository_url: str = DEFAULT_RAG_REPOSITORY,
    revision: str = DEFAULT_RAG_REVISION,
) -> tuple[Path, str]:
    """Clone/check out an immutable RAG revision and return its commit."""
    path = Path(rag_dir)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", repository_url, str(path)],
            check=True,
        )
    current_commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != revision:
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError(
                f"Cannot check out RAG revision {revision}: {path} has local changes"
            )
        exists = subprocess.run(
            ["git", "-C", str(path), "cat-file", "-e", f"{revision}^{{commit}}"],
            capture_output=True,
        ).returncode == 0
        if not exists:
            subprocess.run(
                ["git", "-C", str(path), "fetch", "--depth", "1", "origin", revision],
                check=True,
            )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", revision],
            check=True,
        )
    codex_path = path / "codex.csv"
    index_path = path / "faiss_index"
    if not codex_path.is_file() or not (index_path / "index.faiss").is_file() or not (
        index_path / "index.pkl"
    ).is_file():
        raise FileNotFoundError(
            f"RAG repository at {path} lacks codex.csv or faiss_index artifacts"
        )
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != revision:
        raise RuntimeError(
            f"RAG revision mismatch: expected {revision}, checked out {commit}"
        )
    return path, commit


class PremiseRetriever:
    """Retrieve exact codex provisions before falling back to FAISS."""

    def __init__(self, codex_dataframe, vectorstore: Any):
        required = {"text", "source"}
        if not required.issubset(codex_dataframe.columns):
            raise ValueError("codex.csv must contain text and source columns")
        self._codex = codex_dataframe.loc[:, ["text", "source"]].drop_duplicates()
        self._vectorstore = vectorstore

    @classmethod
    def from_rag_directory(
        cls,
        rag_dir: str | Path,
        *,
        embedding_model: str = "ai-forever/sbert_large_nlu_ru",
        embedding_revision: str = DEFAULT_EMBEDDING_REVISION,
        embedding_device: str = "cpu",
    ) -> "PremiseRetriever":
        """Load trusted cloned codex and FAISS artifacts."""
        import pandas as pd
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        rag_dir = Path(rag_dir).resolve()
        dataframe = pd.read_csv(rag_dir / "codex.csv")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={
                "device": embedding_device,
                "revision": embedding_revision,
            },
        )
        vectorstore = FAISS.load_local(
            rag_dir / "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return cls(dataframe, vectorstore)

    @staticmethod
    def _code_matches(source: str, code: str) -> bool:
        if code == "ГК РФ":
            return source.startswith("ГК РФ ч.") or source.startswith("ГК РФ:")
        return source.startswith(f"{code}:")

    def exact_lookup(self, citation: Citation) -> list[RetrievalRecord]:
        """Return unambiguous exact provisions or an empty list."""
        if not citation.code or not citation.article:
            return []
        article_pattern = re.compile(
            rf"Статья\s+{re.escape(citation.article)}(?!\.\d)(?=\D|$)"
        )
        candidates = self._codex[
            self._codex["source"].map(
                lambda source: self._code_matches(str(source), citation.code)
                and bool(article_pattern.search(str(source)))
            )
        ]
        subprovision = citation.point or citation.part
        if subprovision:
            point_pattern = re.compile(
                rf"п\.\s*{re.escape(subprovision)}(?!\.\d)(?=\D|$)"
            )
            candidates = candidates[
                candidates["source"].map(
                    lambda source: bool(point_pattern.search(str(source)))
                )
            ]
        elif len(candidates) != 1:
            return []
        if candidates.empty:
            return []
        unique = candidates.drop_duplicates(subset=["source", "text"])
        if not subprovision and len(unique) != 1:
            return []
        return [
            RetrievalRecord(
                premise=str(row.text),
                source=str(row.source),
                method="exact",
                rank=index,
                score=None,
                citation=citation,
            )
            for index, row in enumerate(unique.itertuples(index=False), start=1)
        ]

    def retrieve(self, hypothesis: str, *, top_k: int = 20) -> list[RetrievalRecord]:
        """Retrieve premises, enforcing an absolute semantic top-20 limit."""
        citation = extract_citation(hypothesis)
        exact = self.exact_lookup(citation)
        if exact:
            return exact
        bounded_k = max(1, min(int(top_k), 20))
        matches = self._vectorstore.similarity_search_with_score(
            hypothesis, k=bounded_k
        )
        return [
            RetrievalRecord(
                premise=str(document.page_content),
                source=str(document.metadata.get("source", "")),
                method="faiss",
                rank=index,
                score=float(score),
                citation=citation,
            )
            for index, (document, score) in enumerate(matches, start=1)
        ]


def citation_dict(citation: Citation) -> dict[str, str | None]:
    """Return citation fields for tabular audit output."""
    return {f"citation_{key}": value for key, value in asdict(citation).items()}
