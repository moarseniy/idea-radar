from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(default="Новый исследовательский проект", max_length=160)
    goal: str = Field(default="", max_length=4000)
    constraints: str = Field(default="", max_length=6000)
    domain: str = Field(default="Обогащение и металлургия", max_length=160)
    team: list[str] = Field(default_factory=list)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=160)
    goal: str | None = Field(default=None, max_length=4000)
    constraints: str | None = Field(default=None, max_length=6000)
    domain: str | None = Field(default=None, max_length=160)
    team: list[str] | None = None
    settings: dict[str, Any] | None = None


class GenerateRequest(BaseModel):
    count: int = Field(default=5, ge=1, le=10)
    weights: dict[str, float] = Field(default_factory=dict)
    exclusions: list[str] = Field(default_factory=list)
    include_roadmap: bool = True
    research_enabled: bool = False
    research_query: str = Field(default="", max_length=2000)
    research_sources: int = Field(default=6, ge=1, le=12)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=6000)


class FeedbackRequest(BaseModel):
    rating: int | None = Field(default=None, ge=1, le=5)
    outcome: str | None = Field(default=None, max_length=80)
    comment: str = Field(default="", max_length=4000)


class StatusUpdate(BaseModel):
    status: str = Field(max_length=80)


class SampleImportRequest(BaseModel):
    max_files: int = Field(default=12, ge=1, le=20)
    extensions: list[str] = Field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".docx", ".xlsx", ".pdf", ".txt", ".md", ".csv"])
    ocr_languages: list[str] | None = None
