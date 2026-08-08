from pathlib import Path

import pytest

from jura_hypersumm.prompt_sets import load_prompt_set


def test_load_base_prompt_set_without_executing_modules(tmp_path: Path) -> None:
    (tmp_path / "prompt.py").write_text('PROMPT_TEXT = "ternary"\nraise RuntimeError()', encoding="utf-8")
    (tmp_path / "prompt_binary.py").write_text('PROMPT_TEXT_BIN = "binary"', encoding="utf-8")

    prompts = load_prompt_set("base", root=tmp_path)

    assert prompts.ternary == "ternary"
    assert prompts.binary == "binary"


def test_load_named_prompt_pair(tmp_path: Path) -> None:
    (tmp_path / "legal_prompt.py").write_text('PROMPT_TEXT = "t"', encoding="utf-8")
    (tmp_path / "legal_prompt_binary.py").write_text('PROMPT_TEXT_BIN = "b"', encoding="utf-8")
    assert load_prompt_set("legal", root=tmp_path).binary == "b"


def test_prompt_set_rejects_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_prompt_set("../bad", root=tmp_path)
