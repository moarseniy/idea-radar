import json

from scripts import build_negative_dataset as negative
from scripts import extend_negative_dataset as extension


def test_entity_rows_keep_each_repository_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(negative, "ROOT", tmp_path)
    cache = tmp_path / "storage/github_negative_cache"
    cache.mkdir(parents=True)
    first = {
        "name": "AlphaApp", "full_name": "org/AlphaApp",
        "description": "AlphaApp is a personal assistant app",
        "homepage": "https://example.org/alpha", "fork": False, "archived": False,
        "created_at": "2024-01-01T00:00:00Z", "stargazers_count": 100,
        "html_url": "https://github.com/org/AlphaApp",
    }
    second = {
        "name": "BetaApp", "full_name": "org/BetaApp",
        "description": "BetaApp is a chatbot product",
        "homepage": "https://example.org/beta", "fork": False, "archived": False,
        "created_at": "2024-01-02T00:00:00Z", "stargazers_count": 50,
        "html_url": "https://github.com/org/BetaApp",
    }
    (cache / "ai_assistant.json").write_text(json.dumps({"items": [first]}))
    (cache / "chatbot.json").write_text(json.dumps({"items": [second]}))

    rows = negative.entity_rows(limit=2)

    assert len(rows) == 2
    assert [row["source_evidence_excerpt_original"] for row in rows] == [
        first["description"], second["description"],
    ]
    assert [row["primary_source_url"] for row in rows] == [
        first["html_url"], second["html_url"],
    ]
    assert all(row["negative_type"] == "N3b" for row in rows)


def test_extension_skips_existing_and_cross_topic_duplicate(monkeypatch):
    def repository(name, stars):
        return {
            "name": name, "full_name": f"org/{name}",
            "description": f"{name} is a personal finance app",
            "homepage": f"https://example.org/{name}",
            "fork": False, "archived": False,
            "created_at": "2024-01-01T00:00:00Z", "stargazers_count": stars,
            "html_url": f"https://github.com/org/{name}",
        }

    old, alpha, beta = repository("OldApp", 100), repository("AlphaApp", 90), repository("BetaApp", 80)
    monkeypatch.setattr(extension, "SECTORS", [
        ("finance", "topic:finance", "Finance", "Финансы", 1),
        ("health", "topic:health", "Health", "Здоровье", 1),
    ])
    monkeypatch.setattr(extension, "repository_items", lambda key, query: {
        "finance": [old, alpha], "health": [alpha, beta],
    }[key])
    existing = [{"primary_source_url": old["html_url"], "technology_original": old["name"]}]

    selected = extension.choose_repositories(existing, target=2)
    rows = [extension.product_row(repo, spec) for repo, spec in selected]

    assert [repo["name"] for repo, _ in selected] == ["AlphaApp", "BetaApp"]
    assert len({row["primary_source_url"] for row in rows}) == 2
    assert all(row["negative_type"] == "N3b" for row in rows)
    assert all(row["is_technology_flag"] == "" for row in rows)
