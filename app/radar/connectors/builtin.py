"""Коннекторы, которые были до реестра: OpenAlex, гранты, закупки, внедрение, медицина."""
from __future__ import annotations

from app.radar import research_apis as apis
from app.radar.connectors import Connector


def _openalex_enabled(_terms: list[str]) -> bool:
    return apis.openalex_enabled()


CONNECTORS = (
    Connector("OpenAlex", "science", "strict",
              lambda terms, as_of, limit: apis.search_openalex(terms, as_of, limit=limit),
              enabled=_openalex_enabled, language="multi", radar_limit=6),
    Connector("NIH RePORTER", "money", "strict",
              lambda terms, as_of, limit: apis.search_nih_grants(terms, as_of, limit=limit)),
    Connector("CORDIS grants", "money", "strict",
              lambda terms, as_of, limit: apis.search_cordis_grants(terms, as_of, limit=limit),
              language="multi"),
    # GitHub фильтрует по последнему коммиту и отдаёт текущие звёзды (B17),
    # Hugging Face — по дате изменения (B19): для прошлого оба недостоверны.
    Connector("GitHub repositories", "adoption", "none",
              lambda terms, as_of, limit: apis.search_github(terms, as_of, limit=limit)),
    Connector("Hugging Face", "adoption", "none",
              lambda terms, as_of, limit: apis.search_huggingface(terms, as_of, limit=limit),
              enabled=apis.is_ml_topic),
    Connector("TED procurement", "money", "strict",
              lambda terms, as_of, limit: apis.search_ted(terms, as_of, limit=limit),
              language="multi"),
    Connector("ClinicalTrials.gov", "adoption", "post",
              lambda terms, as_of, limit: apis.search_clinicaltrials(terms, as_of, limit=limit),
              enabled=apis.is_health_topic, language="multi"),
    Connector("FDA openFDA", "adoption", "post",
              lambda terms, as_of, limit: apis.search_openfda(terms, as_of, limit=limit),
              enabled=apis.is_health_topic),
)
