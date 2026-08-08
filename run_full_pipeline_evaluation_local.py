"""Re-run validation and full DOCX inference for existing model artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from jura_hypersumm.full_pipeline import run_full_pipeline_evaluation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-source",
        type=Path,
        default=Path("local_artifacts/campaigns/full_pipeline_v1"),
        help="campaign/artifact directory to scan recursively, or a JSON model list",
    )
    parser.add_argument(
        "--rag-source",
        type=Path,
        default=Path("dms-rag"),
        help="legacy dms-rag directory or tuned rag_manifest.json",
    )
    parser.add_argument("--prompt-set", default="base")
    parser.add_argument(
        "--reranker-mode", choices=("none", "bundle", "pretrained"), default="none"
    )
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-revision")
    parser.add_argument("--reranker-trust-remote-code", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--results-dir", type=Path, default=Path("local_results/full_pipeline_evaluation")
    )
    parser.add_argument("--document-batch-size", type=int, default=4)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--candidate-top-k", type=int, default=20)
    parser.add_argument("--final-top-k", type=int, default=20)
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=1024)
    parser.add_argument(
        "--reranker-precision",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--max-retries", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_full_pipeline_evaluation(
        models_source=arguments.models_source,
        rag_source=arguments.rag_source,
        prompt_set=arguments.prompt_set,
        reranker_mode=arguments.reranker_mode,
        reranker_model_id=arguments.reranker_model,
        reranker_revision=arguments.reranker_revision,
        reranker_trust_remote_code=(
            True if arguments.reranker_trust_remote_code else None
        ),
        repo_root=arguments.repo_root,
        results_dir=arguments.results_dir,
        inference_parameters={
            "document_batch_size": arguments.document_batch_size,
            "embedding_device": arguments.embedding_device,
            "candidate_top_k": arguments.candidate_top_k,
            "final_top_k": arguments.final_top_k,
            "reranker_device": arguments.reranker_device,
            "reranker_batch_size": arguments.reranker_batch_size,
            "reranker_max_length": arguments.reranker_max_length,
            "reranker_precision": arguments.reranker_precision,
            "max_retries": arguments.max_retries,
        },
    )
