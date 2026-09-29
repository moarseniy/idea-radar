"""HTTP API сервиса поиска: `uvicorn app.search.server:app --port 8090`.

POST /search   {"text", "as_of", "language", "limit", "lookback_days"} → результаты и аудит провайдеров
GET  /health   провайдеры цепочки и настроены ли они (без ключей)

Ключи провайдеров живут только в этом сервисе. Если задан SEARCH_SERVICE_TOKEN, запросы
к /search должны передавать его в заголовке X-Search-Token.
"""
from __future__ import annotations

import hmac
from datetime import date

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.search.base import SearchQuery
from app.search.service import SearchService

app = FastAPI(title="IDEA Radar Search", version="1.0")
_service: SearchService | None = None


def service() -> SearchService:
    global _service
    if _service is None:
        _service = SearchService()
    return _service


class SearchRequest(BaseModel):
    text: str = Field(min_length=2, max_length=600)
    as_of: date | None = None
    language: str = Field(default="en", max_length=8)
    limit: int = Field(default=10, ge=1, le=50)
    lookback_days: int = Field(default=3 * 365, ge=1, le=20 * 365)


@app.post("/search")
def search(request: SearchRequest, x_search_token: str = Header(default="")) -> dict:
    token = service().settings.service_token
    if token and not hmac.compare_digest(x_search_token, token):
        raise HTTPException(status_code=401, detail="Нужен X-Search-Token")
    return service().search(SearchQuery(**request.model_dump()))


@app.get("/health")
def health() -> dict:
    s = service()
    return {"status": "ok", "mode": s.settings.mode, "providers": s.status()}
