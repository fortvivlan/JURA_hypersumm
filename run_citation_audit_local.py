"""Create a CPU-only audit of rule-based legal citation extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from jura_hypersumm.rag import run_citation_audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", type=Path, default=Path("dms-rag/codex.csv"))
    parser.add_argument("--rag-tests", type=Path, default=Path("rag_tests"))
    parser.add_argument("--dialogue-workbook", type=Path)
    parser.add_argument("--full-workbook", type=Path)
    parser.add_argument("--full-additional-workbook", type=Path)
    parser.add_argument("--test-docx", type=Path, default=Path("test_docx"))
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("local_results/rag/citation_audit"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--routing-scope",
        choices=("all", "rules", "faiss"),
        default="all",
        help="Audit all hypotheses or only those routed to rules/FAISS",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    output = run_citation_audit(
        codex_path=arguments.codex,
        rag_test_dir=arguments.rag_tests,
        dialogue_workbook=arguments.dialogue_workbook,
        full_workbook=arguments.full_workbook,
        full_additional_workbook=arguments.full_additional_workbook,
        test_docx_dir=arguments.test_docx,
        results_dir=arguments.results_dir,
        output_path=arguments.output,
        routing_scope=arguments.routing_scope,
    )
    print(output)
