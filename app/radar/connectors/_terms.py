"""Поисковые фразы для денежных и патентных API и проверка релевантности ответа.

Гранты и закупки почти всегда пустые (B4), потому что в API уходит точная фраза из
длинного названия кандидата: "agent security posture management" не встречается ни в
одном гранте, а "agent security" — встречается. Коннекторы ищут по 2–3 коротким
фразам и затем отсеивают нерелевантное по заголовку и тексту.
"""
from __future__ import annotations

import re

# Слова, которые расширяют фразу, но не сужают поиск: в грантах их пишут как угодно.
GENERIC = {
    "the", "and", "for", "with", "from", "into", "based", "using", "via", "of", "in", "on", "to", "a", "an",
    "technology", "technologies", "system", "systems", "platform", "platforms", "solution", "solutions",
    "management", "approach", "approaches", "application", "applications", "method", "methods",
    "framework", "tool", "tools", "service", "services", "novel", "new", "advanced", "emerging",
    "next-generation", "next", "generation", "early", "stage", "development", "innovative",
}
TOKEN = re.compile(r"[a-z0-9а-яё][a-z0-9а-яё+\-]*", re.IGNORECASE)
CYRILLIC = re.compile(r"[\u0400-\u04ff]")
# Окно близости слов фразы для длинных аннотаций грантов и закупок (см. near).
NEAR = 12


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)]


def key_tokens(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in GENERIC and len(t) > 1]


class ReadyPhrases(list):
    """Уже короткие фразы плана радара: short_phrases берёт их как есть, без вариантов.

    Иначе каждая фраза из трёх слов разворачивается в три варианта и занимает весь лимит
    запросов коннектора, и до API доходит только первая фраза.
    """


def short_phrases(terms: list[str], *, limit: int = 3, max_words: int = 3) -> list[str]:
    """2–3 короткие латинские фразы без общих слов; кириллица пропускается (API англоязычные)."""
    if isinstance(terms, ReadyPhrases):
        ready = [" ".join(key_tokens(term)) for term in terms if not re.search(r"[Ѐ-ӿ]", term)]
        return list(dict.fromkeys(p for p in ready if p))[:limit]
    out: list[str] = []
    for term in terms:
        if re.search(r"[Ѐ-ӿ]", term):
            continue
        words = key_tokens(term)
        if not words:
            continue
        words = words[:max_words]
        candidates = [" ".join(words)]
        # Длинное название даёт ещё и ядра из двух слов: начало ("quantum sensing
        # navigation" → "quantum sensing") и хвост ("automated red teaming" → "red teaming").
        if len(words) > 2:
            candidates += [" ".join(words[:2]), " ".join(words[-2:])]
        out.extend(phrase for phrase in candidates if phrase not in out)
    return out[:limit]


def _stem(word: str) -> str:
    # Грубо, но достаточно для заголовков грантов: agents → agent, batteries → batter(y);
    # русские слова сравниваются по первым 5 буквам (квантовые ~ квантовых).
    if CYRILLIC.search(word):
        return word[:5]
    for suffix in ("ies", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word[:-1] if len(word) > 4 and word.endswith("y") else word


def relevant(text: str, terms: list[str], *, min_share: float = 0.6, window: int | None = None) -> bool:
    """Запись релевантна, если в ней есть большая часть значимых слов хотя бы одной фразы.

    window — слова должны стоять рядом, в пределах window слов текста. Для грантов с
    аннотацией в тысячи слов: "agent" и "identity" в разных абзацах медицинского гранта
    NIH проходили проверку, хотя грант не про идентификацию агентов.
    """
    stems = [_stem(t) for t in _tokens(text)]
    present = set(stems)
    for term in terms:
        words = {_stem(t) for t in key_tokens(term)}
        if not words or len(words & present) / len(words) < min_share:
            continue
        if window is None or any(len(words.intersection(stems[i:i + window])) / len(words) >= min_share
                                 for i in range(max(1, len(stems) - window + 1))):
            return True
    return False


def near(text: str, terms: list[str]) -> bool:
    """Проверка для длинных аннотаций: слова названия или запрошенной короткой фразы рядом."""
    return relevant(text, [*terms, *short_phrases(terms)], window=NEAR)
