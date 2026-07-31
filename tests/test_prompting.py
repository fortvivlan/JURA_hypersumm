from prompt import PROMPT_TEXT
from prompt_binary import PROMPT_TEXT_BIN

from jura_hypersumm.prompting import (
    build_generation_prompt,
    build_training_texts,
    parse_generated_label,
)


class PlainTokenizer:
    chat_template = None


def test_prompts_use_each_imported_constant_once() -> None:
    tokenizer = PlainTokenizer()
    ternary = build_generation_prompt(tokenizer, "premise", "hypothesis", "ternary")
    binary = build_generation_prompt(tokenizer, "premise", "hypothesis", "binary")

    assert ternary.count(PROMPT_TEXT) == 1
    assert binary.count(PROMPT_TEXT_BIN) == 1


def test_training_text_contains_only_one_system_prompt() -> None:
    prompt, full = build_training_texts(
        PlainTokenizer(), "premise", "hypothesis", "no", "binary"
    )
    assert prompt.count(PROMPT_TEXT_BIN) == 1
    assert full.count(PROMPT_TEXT_BIN) == 1
    assert full.endswith("Ответ: no")


def test_parse_generated_labels_is_strict_and_removes_thinking() -> None:
    assert (
        parse_generated_label(
            "<think>maybe entailment</think> not mentioned", "ternary"
        )
        == "not mentioned"
    )
    assert parse_generated_label("not", "binary") is None
    assert parse_generated_label("no", "binary") == "no"
    assert parse_generated_label("no contradiction", "binary") is None
