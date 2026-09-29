import app.registry as registry_module
from app.registry import (
    classify,
    is_high_trust,
    is_press_release,
    is_unverifiable,
    load,
    registry_table,
    supports_signal,
)


def test_registry_loads_and_every_domain_has_known_type():
    data = load()
    assert data["domains"] and all(d["type"] in data["types"] for d in data["domains"])
    assert len({d["domain"] for d in data["domains"]}) == len(data["domains"])
    assert len(data["domains"]) >= 300


def test_classify_domains_subdomains_and_patterns():
    assert classify("https://www.reuters.com/tech/x").trust == "high"
    assert classify("https://blog.medium.com/p").type == "blog"
    assert classify("https://spectrum.ieee.org/a").type == "tech_media"  # точный поддомен важнее ieee.org
    assert classify("https://www.nist.gov/news").type == "standards_body"
    assert classify("https://energy.gov/article").type == "institution"
    assert classify("https://acme.com/press-release/2026").type == "press_release"
    assert classify("https://www.nasa.gov/press-release/mission").type == "press_release"
    assert classify("https://www.gov.za/department/research").trust == "high"
    assert classify("https://www.ema.europa.eu/en").name == "European Medicines Agency"
    assert classify("https://www.csiro.au/en").type == "institution"
    assert classify("https://www.jpo.go.jp/").type == "patent"
    assert classify("https://www.katadata.co.id/").type == "industry_media"
    unknown = classify("https://some-startup.io/blog")
    assert unknown.type == "website" and unknown.trust == "unverified" and unknown.matched is None


def test_feature_helpers():
    assert is_press_release("prnewswire.com") and not is_press_release("reuters.com")
    assert is_high_trust("nature.com") and not is_high_trust("medium.com")
    assert is_unverifiable("t.me") and is_unverifiable("") and not is_unverifiable("habr.com")


def test_primary_indicators_cannot_confirm_signal_alone():
    ok, notes = supports_signal([{"url": "https://x.com/a"}, {"url": "https://prnewswire.com/b"}])
    assert not ok and "первичными индикаторами" in notes[0]
    ok, notes = supports_signal([{"url": "https://techcrunch.com/a"}])
    assert ok and any("одного независимого" in n for n in notes)
    ok, notes = supports_signal([{"url": "https://techcrunch.com/a"}, {"url": "https://www.cbr.ru/b"}])
    assert ok and notes == []


def test_registry_table_for_ui():
    table = registry_table()
    assert {t["type"] for t in table["types"]} >= {"social", "blog", "aggregator"}
    assert all(t["primary_only"] for t in table["types"] if t["type"] in {"social", "blog", "aggregator"})


def test_user_domain_overrides_are_persisted_and_used_for_classification(tmp_path, monkeypatch):
    overrides_path = tmp_path / "source_registry_overrides.json"
    monkeypatch.setattr(registry_module, "OVERRIDES_PATH", overrides_path)
    registry_module._load_registry.cache_clear()
    try:
        table = registry_module.save_domain("www.example.test", "Example Tech", "blog", "en")
        entry = next(d for d in table["domains"] if d["domain"] == "example.test")
        assert entry["customized"] is True
        assert entry["trust"] == "low" and entry["primary_only"] is True
        assert registry_module.classify("https://updates.example.test/post").type == "blog"

        registry_module.save_domain("example.test", "Example News", "tech_media", "en")
        assert registry_module.classify("https://example.test/story").type == "tech_media"
        assert registry_module.delete_domain("example.test") is True
        assert registry_module.classify("https://example.test/story").type == "website"
        assert overrides_path.exists()
    finally:
        registry_module._load_registry.cache_clear()
