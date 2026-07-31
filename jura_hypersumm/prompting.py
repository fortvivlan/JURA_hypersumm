"""Prompt construction and strict generated-label parsing."""

from __future__ import annotations

import re
from typing import Any

from prompt import PROMPT_TEXT
from prompt_binary import PROMPT_TEXT_BIN

from .common import LABELS_BY_TASK, Task


def prompt_for_task(task: Task) -> str:
    """Return the imported canonical prompt for a task."""
    return PROMPT_TEXT_BIN if task == "binary" else PROMPT_TEXT


def build_messages(
    premise: str,
    hypothesis: str,
    task: Task,
    *,
    assistant_label: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages without duplicating the system prompt."""
    messages = [
        {"role": "system", "content": prompt_for_task(task)},
        {
            "role": "user",
            "content": f"Предпосылка: {premise}\nГипотеза: {hypothesis}",
        },
    ]
    if assistant_label is not None:
        messages.append({"role": "assistant", "content": assistant_label})
    return messages


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    """Apply a tokenizer chat template, with a plain-text fallback."""
    if getattr(tokenizer, "chat_template", None):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    system = messages[0]["content"]
    user = messages[1]["content"]
    if len(messages) == 3:
        return f"{system}\n\n{user}\nОтвет: {messages[2]['content']}"
    return f"{system}\n\n{user}\nОтвет:"


def build_generation_prompt(
    tokenizer: Any, premise: str, hypothesis: str, task: Task
) -> str:
    """Build one inference prompt from the canonical imported task prompt."""
    return apply_chat_template(
        tokenizer,
        build_messages(premise, hypothesis, task),
        add_generation_prompt=True,
    )


def build_training_texts(
    tokenizer: Any,
    premise: str,
    hypothesis: str,
    label: str,
    task: Task,
) -> tuple[str, str]:
    """Return prompt-only and prompt-plus-label strings for response-only loss."""
    if label not in LABELS_BY_TASK[task]:
        raise ValueError(f"Invalid {task} label: {label!r}")
    prompt = build_generation_prompt(tokenizer, premise, hypothesis, task)
    full = apply_chat_template(
        tokenizer,
        build_messages(premise, hypothesis, task, assistant_label=label),
        add_generation_prompt=False,
    )
    return prompt, full


def parse_generated_label(text: str, task: Task) -> str | None:
    """Parse exactly one supported label from generated model text."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    normalized = cleaned.lower().strip()
    if task == "ternary":
        patterns = (
            ("not mentioned", r"\bnot\s+mentioned\b"),
            ("contradiction", r"\bcontradiction\b"),
            ("entailment", r"\bentailment\b"),
        )
    else:
        patterns = (
            ("contradiction", r"\bcontradiction\b"),
            ("no", r"\bno\b"),
        )
    matches = [label for label, pattern in patterns if re.search(pattern, normalized)]
    return matches[0] if len(matches) == 1 else None
