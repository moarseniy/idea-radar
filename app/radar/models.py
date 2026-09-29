from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchRequest(StrictModel):
    query: str = Field(min_length=3, max_length=1200)
    as_of: date = Field(default_factory=date.today)
    limit: int = Field(default=15, ge=1, le=15)
    directions: list[str] = Field(default_factory=list, max_length=7)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 3:
            raise ValueError("Введите технологическое направление (не менее трёх символов).")
        return value

    @field_validator("directions")
    @classmethod
    def clean_directions(cls, values: list[str]) -> list[str]:
        clean, seen = [], set()
        for value in values:
            value = " ".join(value.split())
            key = value.casefold()
            if len(value) < 3 or len(value) > 240:
                raise ValueError("Каждое выбранное направление должно содержать 3–240 символов.")
            if key not in seen:
                clean.append(value)
                seen.add(key)
        return clean

    @field_validator("as_of")
    @classmethod
    def past_date(cls, value: date) -> date:
        if value > date.today():  # noqa: DTZ011 — дата поиска использует локальный календарный день
            raise ValueError("Дата оценки не может быть в будущем.")
        return value


class RegistryDomainInput(StrictModel):
    domain: str = Field(min_length=3, max_length=253)
    name: str = Field(min_length=1, max_length=120)
    type: str = Field(min_length=2, max_length=40)
    lang: str = Field(default="en", min_length=2, max_length=8)


class LocalizedQuery(StrictModel):
    language: str
    query: str


class LocalizedQueryBatch(StrictModel):
    queries: list[LocalizedQuery]


class Branch(StrictModel):
    topic: str
    query_ru: str
    query_en: str
    localized_queries: list[LocalizedQuery] = Field(default_factory=list)


class SearchPlan(StrictModel):
    interpretation: str
    branches: list[Branch]
    query_language: str = "und"


class ScopeChatMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1200)


class ScopeSuggestion(StrictModel):
    title: str = Field(min_length=3, max_length=180)
    description: str = Field(max_length=320)


class ScopeChatRequest(StrictModel):
    query: str = Field(min_length=3, max_length=1200)
    message: str = Field(default="", max_length=1200)
    history: list[ScopeChatMessage] = Field(default_factory=list, max_length=12)
    selected_directions: list[str] = Field(default_factory=list, max_length=7)


class ScopeChatResponse(StrictModel):
    reply: str = Field(min_length=1, max_length=1200)
    suggestions: list[ScopeSuggestion] = Field(default_factory=list, max_length=4)


class Candidate(StrictModel):
    id: str
    title: str
    application: str
    domain: str
    summary: str
    source_ids: list[str]
    verification_query: str


class CandidateBatch(StrictModel):
    candidates: list[Candidate]


class Evidence(StrictModel):
    source_id: str
    quote: str
    claim: str
    relation: Literal["supports", "contradicts", "context"]
    event_date: str | None


class Finding(StrictModel):
    text: str
    source_ids: list[str]


class Predictor(StrictModel):
    name: Literal["early_stage", "novelty", "momentum", "evidence", "maturity"]
    value: Literal["strong", "partial", "unknown", "contradictory"]
    explanation: str
    source_ids: list[str]


class Assessment(StrictModel):
    candidate_id: str
    verdict: Literal["weak_signal", "mature", "hype_or_noise", "irrelevant", "insufficient_evidence"]
    reason: str
    stage: str
    description: Finding
    why_now: Finding
    why_early: Finding
    advantage: Finding
    case: Finding
    case_kind: Literal["observed", "proposed", "unknown"]
    business_use: str
    limitations: list[str]
    predictors: list[Predictor]
    evidence: list[Evidence]


class AssessmentBatch(StrictModel):
    assessments: list[Assessment]
