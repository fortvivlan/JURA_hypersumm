"""Recall-only evaluation of the complete rule-first RAG pipeline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from ..documents import (
    extract_operative_section,
    read_docx_text,
    split_russian_sentences,
    textcheck,
)
from ..retrieval import Citation, PremiseRetriever, extract_citations, parse_codex_source
from .artifacts import RagBundle, load_rag_bundle
from .reranking import Reranker, score_and_sort_records

RECALL_CUTOFFS = (1, 5, 10, 20)


def normalize_sentence(value: object) -> str:
    """Normalize a sentence for conservative exact workbook/DOCX alignment."""
    return " ".join(str(value or "").replace("\xa0", " ").split()).casefold()


def _citation_key(citation: Citation) -> tuple[str, str, str, str]:
    return tuple(
        " ".join(str(value or "").split()).casefold()
        for value in (citation.code, citation.article, citation.part, citation.point)
    )  # type: ignore[return-value]


def _parse_gold(value: str) -> Citation:
    citations = extract_citations(value)
    if len(citations) != 1 or not citations[0].article:
        raise ValueError(f"Could not parse one gold citation: {value!r}")
    return citations[0]


def read_rag_workbook(path: str | Path):
    """Read a headerless curated workbook and group rows by sentence."""
    import pandas as pd

    source = Path(path)
    frame = pd.read_excel(
        source, header=None, dtype=str, keep_default_na=False, engine="openpyxl"
    )
    if frame.shape[1] != 3:
        raise ValueError(f"{source} must have exactly three columns")
    frame.columns = ["sentence", "gold_reference", "gold_text"]
    frame["normalized_sentence"] = frame["sentence"].map(normalize_sentence)
    if frame["normalized_sentence"].eq("").any():
        raise ValueError(f"{source} contains a blank sentence")
    rows = []
    for normalized, group in frame.groupby("normalized_sentence", sort=False):
        references = [value for value in group["gold_reference"] if value.strip()]
        citations: dict[tuple[str, str, str, str], Citation] = {}
        for reference in references:
            citation = _parse_gold(reference)
            citations[_citation_key(citation)] = citation
        rows.append(
            {
                "normalized_sentence": normalized,
                "sentence": group.iloc[0]["sentence"],
                "gold_citations": tuple(citations.values()),
                "gold_references": tuple(dict.fromkeys(references)),
                "gold_texts": tuple(
                    dict.fromkeys(value for value in group["gold_text"] if value.strip())
                ),
            }
        )
    return rows


def _document_sentences(directory: Path) -> dict[str, set[str]]:
    aligned: dict[str, set[str]] = defaultdict(set)
    for document in sorted(directory.glob("*.docx")):
        section = extract_operative_section(read_docx_text(document))
        if section is None:
            continue
        for sentence in split_russian_sentences(section):
            if not textcheck(sentence):
                aligned[normalize_sentence(sentence)].add(document.name)
    return aligned


def _default_document_directory(workbook: Path, test_docx_dir: Path) -> Path:
    if "тестовые_доки_диалог" in workbook.name.casefold():
        return test_docx_dir / "Dialogue"
    return test_docx_dir / "Full"


def _retrieved_citation(source: str) -> Citation | None:
    parsed = parse_codex_source(source)
    if parsed is not None:
        return parsed
    citations = extract_citations(source)
    return citations[0] if len(citations) == 1 else None


def calculate_recall_rows(
    query_rows,
    *,
    rag_name: str,
    workbook_name: str,
    cutoffs: Sequence[int] = RECALL_CUTOFFS,
    candidate_top_k: int | None = None,
):
    """Calculate candidate-pool and final query/article recall rows."""
    rows = []
    for branch in ("all", "exact", "faiss"):
        selected = query_rows if branch == "all" else [r for r in query_rows if r["method"] == branch]
        gold_total = sum(len(row["gold_keys"]) for row in selected)
        if candidate_top_k is not None:
            query_hits = sum(
                bool(row["gold_keys"] & set(row.get("candidate_keys", row["retrieved_keys"])))
                for row in selected
            )
            article_hits = sum(
                len(row["gold_keys"] & set(row.get("candidate_keys", row["retrieved_keys"])))
                for row in selected
            )
            rows.append(
                {
                    "rag_version": rag_name,
                    "workbook": workbook_name,
                    "branch": branch,
                    "stage": "candidate",
                    "cutoff": candidate_top_k,
                    "queries": len(selected),
                    "gold_articles": gold_total,
                    "query_recall": query_hits / len(selected) if selected else None,
                    "article_micro_recall": article_hits / gold_total if gold_total else None,
                }
            )
        for cutoff in cutoffs:
            query_hits = sum(bool(row["gold_keys"] & set(row["retrieved_keys"][:cutoff])) for row in selected)
            article_hits = sum(len(row["gold_keys"] & set(row["retrieved_keys"][:cutoff])) for row in selected)
            rows.append(
                {
                    "rag_version": rag_name,
                    "workbook": workbook_name,
                    "branch": branch,
                    "stage": "final",
                    "cutoff": cutoff,
                    "queries": len(selected),
                    "gold_articles": gold_total,
                    "query_recall": query_hits / len(selected) if selected else None,
                    "article_micro_recall": article_hits / gold_total if gold_total else None,
                }
            )
    return rows


def _make_retriever(bundle: RagBundle, device: str) -> PremiseRetriever:
    return PremiseRetriever.from_components(
        bundle.codex_path,
        bundle.index_dir,
        embedding_model=bundle.embedding_model,
        embedding_revision=bundle.embedding_revision,
        embedding_device=device,
        normalize_embeddings=bundle.normalize_embeddings,
    )


def _comparison_deltas(scores, bundles, variant_names, has_reranker: bool):
    """Return named embedding, reranker, and combined recall deltas."""
    import pandas as pd

    if len(bundles) != 2 or scores.empty:
        return pd.DataFrame()
    baseline, tuned = bundles
    base_no = variant_names[(baseline.name, False)]
    tuned_no = variant_names[(tuned.name, False)]
    comparisons = [("embedding_without_reranker", base_no, tuned_no)]
    if has_reranker:
        base_yes = variant_names[(baseline.name, True)]
        tuned_yes = variant_names[(tuned.name, True)]
        comparisons.extend(
            [
                ("reranker_on_baseline", base_no, base_yes),
                ("reranker_on_tuned", tuned_no, tuned_yes),
                ("combined_vs_unchanged_baseline", base_no, tuned_yes),
            ]
        )
    keys = ["workbook", "branch", "stage", "cutoff"]
    metrics = ["query_recall", "article_micro_recall"]
    frames = []
    for name, left_name, right_name in comparisons:
        left = scores[scores.rag_version == left_name].set_index(keys)
        right = scores[scores.rag_version == right_name].set_index(keys)
        delta = right[metrics].subtract(left[metrics]).reset_index()
        delta.insert(0, "comparison", name)
        delta.insert(1, "from_variant", left_name)
        delta.insert(2, "to_variant", right_name)
        frames.append(delta)
    return pd.concat(frames, ignore_index=True)


def run_rag_evaluation(
    rag_sources: Sequence[str | Path],
    *,
    rag_test_dir: str | Path = "rag_tests",
    test_docx_dir: str | Path = "test_docx",
    results_dir: str | Path = "local_results/rag/evaluation",
    embedding_device: str = "cpu",
    retriever_factory: Callable[[RagBundle, str], PremiseRetriever] | None = None,
    candidate_top_k: int = 20,
    final_top_k: int = 20,
    recall_cutoffs: Sequence[int] = RECALL_CUTOFFS,
    reranker: Reranker | None = None,
    compare_reranking: bool = True,
):
    """Evaluate retrieval versions with matched no-reranker/reranker candidates."""
    import pandas as pd

    if candidate_top_k <= 0 or final_top_k <= 0:
        raise ValueError("candidate_top_k and final_top_k must be positive")
    if final_top_k > candidate_top_k:
        raise ValueError("final_top_k cannot exceed candidate_top_k")
    cutoffs = sorted(
        {int(value) for value in recall_cutoffs if 0 < int(value) <= final_top_k}
        | {final_top_k}
    )
    bundles = [load_rag_bundle(source) for source in rag_sources]
    workbooks = sorted(
        path for path in Path(rag_test_dir).glob("*.xlsx") if not path.name.startswith("~$")
    )
    if not workbooks:
        raise FileNotFoundError(f"No RAG workbooks found in {rag_test_dir}")
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    audit_rows: list[dict] = []
    unmatched_rows: list[dict] = []
    zero_gold_rows: list[dict] = []
    make_retriever = retriever_factory or _make_retriever
    variant_names: dict[tuple[str, bool], str] = {}
    for bundle in bundles:
        retriever = make_retriever(bundle, embedding_device)
        reranker_name = (
            reranker.model_id.rstrip("/").split("/")[-1] if reranker is not None else "none"
        )
        variant_names[(bundle.name, False)] = f"{bundle.name}__no_reranker"
        variant_names[(bundle.name, True)] = f"{bundle.name}__{reranker_name}"
        for workbook in workbooks:
            sentence_map = _document_sentences(
                _default_document_directory(workbook, Path(test_docx_dir))
            )
            scored_by_variant: dict[str, list[dict]] = defaultdict(list)
            for row in read_rag_workbook(workbook):
                common = {
                    "rag_version": bundle.name,
                    "workbook": workbook.name,
                    "sentence": row["sentence"],
                    "documents": " | ".join(sorted(sentence_map.get(row["normalized_sentence"], set()))),
                    "gold_references": " | ".join(row["gold_references"]),
                }
                if row["normalized_sentence"] not in sentence_map:
                    unmatched_rows.append(common)
                    continue
                if not row["gold_citations"]:
                    zero_gold_rows.append(common)
                    continue
                outcome = retriever.retrieve_with_details(
                    row["sentence"],
                    top_k=candidate_top_k,
                    final_top_k=final_top_k,
                )
                gold_keys = {_citation_key(value) for value in row["gold_citations"]}
                method = outcome.results[0].method if outcome.results else "none"
                candidate_citations = [
                    _retrieved_citation(record.source) for record in outcome.candidates
                ]
                candidate_keys = [
                    _citation_key(value) for value in candidate_citations if value is not None
                ]
                configurations = [(False, outcome.results)]
                reranked_candidates = ()
                if reranker is not None and compare_reranking:
                    reranked_candidates = (
                        score_and_sort_records(
                            row["sentence"],
                            outcome.candidates,
                            reranker,
                        )
                        if method == "faiss"
                        else outcome.results
                    )
                    configurations.append(
                        (True, reranked_candidates[:final_top_k])
                    )
                for uses_reranker, records in configurations:
                    variant = variant_names[(bundle.name, uses_reranker)]
                    final_citations = [
                        _retrieved_citation(record.source) for record in records
                    ]
                    final_keys = [
                        _citation_key(value)
                        for value in final_citations
                        if value is not None
                    ]
                    scored_by_variant[variant].append(
                        {
                            "gold_keys": gold_keys,
                            "candidate_keys": candidate_keys,
                            "retrieved_keys": final_keys,
                            "method": method,
                        }
                    )
                    final_by_initial_rank = {
                        int(record.initial_rank or record.rank): record for record in records
                    }
                    scored_by_initial_rank = {
                        int(record.initial_rank or record.rank): record
                        for record in reranked_candidates
                    }
                    for candidate, citation in zip(
                        outcome.candidates, candidate_citations
                    ):
                        final_record = final_by_initial_rank.get(
                            int(candidate.initial_rank or candidate.rank)
                        )
                        scored_record = scored_by_initial_rank.get(
                            int(candidate.initial_rank or candidate.rank)
                        )
                        audit_rows.append(
                            {
                                **common,
                                "rag_variant": variant,
                                "reranker_enabled": uses_reranker,
                                "method": candidate.method,
                                "initial_rank": candidate.initial_rank,
                                "retrieval_score": candidate.score,
                                "final_rank": final_record.rank if final_record else None,
                                "reranker_score": (
                                    scored_record.reranker_score
                                    if uses_reranker and scored_record
                                    else None
                                ),
                                "retained": final_record is not None,
                                "retrieved_source": candidate.source,
                                "retrieved_citation": asdict(citation) if citation else None,
                                "is_gold": citation is not None
                                and _citation_key(citation) in gold_keys,
                            }
                        )
            for variant, scored in scored_by_variant.items():
                rows = calculate_recall_rows(
                    scored,
                    rag_name=variant,
                    workbook_name=workbook.name,
                    cutoffs=cutoffs,
                    candidate_top_k=candidate_top_k,
                )
                uses_reranker = variant != variant_names[(bundle.name, False)]
                for score_row in rows:
                    score_row["embedding_version"] = bundle.name
                    score_row["reranker_enabled"] = uses_reranker
                    score_row["candidate_top_k"] = candidate_top_k
                    score_row["final_top_k"] = final_top_k
                summary_rows.extend(rows)
    scores = pd.DataFrame(summary_rows)
    deltas = _comparison_deltas(scores, bundles, variant_names, reranker is not None)
    tables = {
        "recall": scores,
        "deltas": deltas,
        "retrieval_audit": pd.DataFrame(audit_rows),
        "unmatched": pd.DataFrame(unmatched_rows),
        "zero_gold": pd.DataFrame(zero_gold_rows),
    }
    scores.to_csv(output / "rag_recall.csv", index=False, encoding="utf-8")
    with pd.ExcelWriter(output / "rag_recall.xlsx", engine="openpyxl") as writer:
        for sheet, table in tables.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)
    return scores
