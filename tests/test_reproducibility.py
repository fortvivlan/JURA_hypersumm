import os
import random
import re

import numpy as np

from jura_hypersumm.common import (
    DEFAULT_BERT_REVISION,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_RAG_REVISION,
    MODEL_SPECS,
    configure_reproducibility,
    resolve_huggingface_revision,
    source_tree_sha256,
)


def test_reproducibility_configuration_repeats_rng_sequences() -> None:
    configure_reproducibility(123, deterministic=True)
    first = (random.random(), np.random.random())
    configure_reproducibility(123, deterministic=True)
    second = (random.random(), np.random.random())

    assert first == second
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_pinned_revisions_are_immutable_hashes() -> None:
    assert DEFAULT_BERT_REVISION == DEFAULT_EMBEDDING_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", DEFAULT_BERT_REVISION)
    assert re.fullmatch(r"[0-9a-f]{40}", DEFAULT_RAG_REVISION)
    for spec in MODEL_SPECS:
        if spec.alias != "llama":
            assert spec.revision is not None
            assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)


def test_exact_huggingface_revision_does_not_need_network() -> None:
    assert (
        resolve_huggingface_revision("any/model", DEFAULT_BERT_REVISION)
        == DEFAULT_BERT_REVISION
    )


def test_source_tree_fingerprint_is_stable() -> None:
    first = source_tree_sha256()
    second = source_tree_sha256()
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)
