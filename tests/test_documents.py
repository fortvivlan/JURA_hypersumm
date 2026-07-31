from jura_hypersumm.documents import (
    extract_operative_section,
    split_russian_sentences,
)


def test_extract_operative_section_uses_final_spaced_marker() -> None:
    text = "Вводная часть. ПОСТАНОВИЛ: старое. П О С Т А Н О В И Л : Назначить штраф."
    assert extract_operative_section(text) == "Назначить штраф."


def test_extract_operative_section_returns_none_when_missing_or_empty() -> None:
    assert extract_operative_section("РЕШИЛ: отказать") is None
    assert extract_operative_section("ПОСТАНОВИЛ:") is None


def test_split_russian_sentences() -> None:
    assert split_russian_sentences("Назначить штраф. Жалоба может быть подана!") == [
        "Назначить штраф.",
        "Жалоба может быть подана!",
    ]
