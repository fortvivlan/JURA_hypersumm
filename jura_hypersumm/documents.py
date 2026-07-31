"""Court-decision parsing helpers."""

from __future__ import annotations

import re
from pathlib import Path

POSTANOVIL_PATTERN = re.compile(
    r"П\s*О\s*С\s*Т\s*А\s*Н\s*О\s*В\s*И\s*Л\s*[:.]?",
    flags=re.IGNORECASE,
)
IRRELEVANT_SENTENCE_MARKERS = (
    "реквизит",
    "ре...изит",
    "квитанци",
    "судья",
)


def read_docx_text(path: str | Path) -> str:
    """Read paragraphs and table cells from a DOCX file."""
    from docx import Document

    document = Document(Path(path))
    blocks = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
    for table in document.tables:
        for row in table.rows:
            blocks.extend(cell.text for cell in row.cells if cell.text)
    return "\n".join(blocks)


def extract_operative_section(text: str) -> str | None:
    """Return text after the final ПОСТАНОВИЛ marker, or ``None`` if absent."""
    matches = list(POSTANOVIL_PATTERN.finditer(text))
    if not matches:
        return None
    section = text[matches[-1].end() :].strip()
    return section or None


def split_russian_sentences(text: str) -> list[str]:
    """Split nonempty Russian sentences with razdel."""
    from razdel import sentenize

    return [sentence.text.strip() for sentence in sentenize(text) if sentence.text.strip()]


def textcheck(text: str) -> bool:
    """Return whether a sentence is irrelevant to codex contradiction checks."""
    normalized = text.lower()
    return any(marker in normalized for marker in IRRELEVANT_SENTENCE_MARKERS)
