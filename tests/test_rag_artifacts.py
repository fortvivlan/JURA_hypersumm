from pathlib import Path

from jura_hypersumm.rag.artifacts import load_rag_bundle, write_rag_manifest


def test_manifest_round_trip_resolves_local_reranker(tmp_path: Path) -> None:
    codex = tmp_path / "codex.csv"
    codex.write_text("text,source\ntext,source\n", encoding="utf-8")
    embedding = tmp_path / "embedding_model"
    embedding.mkdir()
    index = tmp_path / "faiss_index"
    index.mkdir()
    (index / "index.faiss").write_bytes(b"faiss")
    (index / "index.pkl").write_bytes(b"metadata")
    reranker = tmp_path / "reranker_model"
    reranker.mkdir()

    manifest = write_rag_manifest(
        tmp_path,
        experiment_id="legal-rag",
        codex_path=codex,
        embedding_model_dir=embedding,
        index_dir=index,
        metadata={},
        reranker={
            "mode": "finetuned",
            "model": "reranker_model",
            "local": True,
            "revision": None,
            "trust_remote_code": True,
            "max_length": 768,
        },
    )

    bundle = load_rag_bundle(manifest)

    assert bundle.reranker is not None
    assert bundle.reranker.mode == "finetuned"
    assert bundle.reranker.model == str(reranker)
    assert bundle.reranker.max_length == 768


def test_legacy_manifest_without_reranker_remains_loadable(tmp_path: Path) -> None:
    codex = tmp_path / "codex.csv"
    codex.write_text("text,source\ntext,source\n", encoding="utf-8")
    embedding = tmp_path / "embedding_model"
    embedding.mkdir()
    index = tmp_path / "faiss_index"
    index.mkdir()
    (index / "index.faiss").write_bytes(b"faiss")
    (index / "index.pkl").write_bytes(b"metadata")

    manifest = write_rag_manifest(
        tmp_path,
        experiment_id="legal-rag",
        codex_path=codex,
        embedding_model_dir=embedding,
        index_dir=index,
        metadata={},
    )

    assert load_rag_bundle(manifest).reranker is None
