"""Excel result serialization shared by all workflows."""

from __future__ import annotations

import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .common import DEFAULT_RESULTS_DIR, json_value


def concatenate_tables(tables: list[Any]):
    """Concatenate DataFrames, returning an empty DataFrame for no inputs."""
    import pandas as pd

    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def metadata_table(metadata: Mapping[str, Any]):
    """Convert run metadata to a two-column DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        [{"key": key, "value": json_value(value)} for key, value in metadata.items()]
    )


def write_results_workbook(
    workflow_name: str,
    tables: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """Write one timestamped multi-sheet XLSX result workbook."""
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{workflow_name}_{timestamp}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            if table is None:
                table = pd.DataFrame()
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        metadata_table(metadata).to_excel(writer, sheet_name="run_metadata", index=False)
    return path


def display_scores(scores) -> None:
    """Display a score table in notebooks, with a plain terminal fallback."""
    try:
        from IPython.display import display

        display(scores)
    except ModuleNotFoundError:
        print(scores.to_string(index=False))


def _safe_artifact_name(value: str, *, limit: int = 80) -> str:
    """Return a readable filename component safe on common operating systems."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" ._")
    return (cleaned or "document")[:limit]


def _article_number(source: object, citation_article: object = None) -> str:
    """Extract a dotted article number from retrieved-source metadata."""
    match = re.search(r"(?i)Статья\s+([0-9]+(?:\.[0-9]+)*)", str(source or ""))
    if match:
        return match.group(1)
    if citation_article is None or str(citation_article) == "nan":
        return ""
    return str(citation_article)


def write_document_review_package(
    workflow_name: str,
    document_pairs,
    *,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path | None:
    """Create a ZIP of per-document model and top-1 RAG review workbooks.

    Every document/task model workbook contains all classified premise pairs
    and blank specialist fields for later scoring. Every document RAG workbook
    contains exactly one top-ranked retrieved article per processed sentence.
    ``None`` is returned when no document pairs were produced.
    """
    import pandas as pd

    if document_pairs is None or document_pairs.empty:
        return None
    required = {
        "document",
        "task",
        "hypothesis_id",
        "sentence_index",
        "hypothesis",
        "premise",
        "source",
        "retrieval_rank",
        "prediction",
    }
    missing = sorted(required - set(document_pairs.columns))
    if missing:
        raise ValueError(
            "Document pair table lacks review-package columns: " + ", ".join(missing)
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = output_dir / f"{_safe_artifact_name(workflow_name)}_document_review_{timestamp}.zip"
    with tempfile.TemporaryDirectory(
        prefix="jura_review_", dir=output_dir
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        ordered = document_pairs.sort_values(
            ["document", "task", "sentence_index", "retrieval_rank"],
            kind="stable",
        )
        for document_name, document_rows in ordered.groupby("document", sort=False):
            document_slug = _safe_artifact_name(Path(str(document_name)).stem)
            for task, task_rows in document_rows.groupby("task", sort=False):
                model_review = task_rows.loc[
                    :, ["hypothesis", "premise", "prediction"]
                ].rename(columns={"prediction": "model_prediction"})
                model_review["expert_label"] = ""
                model_review["expert_comment"] = ""
                model_path = temporary / (
                    f"{document_slug}_{_safe_artifact_name(str(task))}_model_predictions.xlsx"
                )
                model_review.to_excel(
                    model_path, sheet_name="model_predictions", index=False
                )

            # Retrieval is identical for tasks using the same document. Keep
            # the best-ranked candidate once per original sentence.
            top_retrieval = (
                document_rows.sort_values(
                    ["sentence_index", "retrieval_rank", "task"], kind="stable"
                )
                .drop_duplicates(subset=["hypothesis_id"], keep="first")
                .copy()
            )
            top_retrieval["article_number"] = [
                _article_number(source, citation)
                for source, citation in zip(
                    top_retrieval["source"],
                    top_retrieval.get(
                        "citation_article",
                        pd.Series([None] * len(top_retrieval), index=top_retrieval.index),
                    ),
                )
            ]
            rag_review = top_retrieval.loc[
                :, ["hypothesis", "article_number", "premise"]
            ].rename(
                columns={"hypothesis": "sentence", "premise": "article_text"}
            )
            rag_path = temporary / f"{document_slug}_rag_retrieval.xlsx"
            rag_review.to_excel(rag_path, sheet_name="rag_retrieval", index=False)

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for workbook in sorted(temporary.glob("*.xlsx")):
                archive.write(workbook, arcname=workbook.name)
    return archive_path
