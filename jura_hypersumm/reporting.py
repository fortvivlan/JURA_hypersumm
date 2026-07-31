"""Excel result serialization shared by all workflows."""

from __future__ import annotations

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
