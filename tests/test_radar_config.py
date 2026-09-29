from dataclasses import replace

from app.radar.config import RadarSettings, load_radar_settings
from app.radar.pipeline import make_provider
from app.radar.provider_openrouter import OpenRouterProvider


def test_radar_defaults_to_openrouter_and_needs_only_its_key():
    settings = RadarSettings()
    assert settings.model == "openai/gpt-4.1"
    assert "OPENROUTER_API_KEY" in settings.configuration_error()
    ready = replace(settings, openrouter_api_key="test-key")
    assert ready.configuration_error() is None


def test_radar_has_no_direct_api_provider():
    settings = replace(RadarSettings(), openrouter_api_key="test-key")
    provider = make_provider(settings, deadline=10**12)
    try:
        assert isinstance(provider, OpenRouterProvider)
        assert provider.client.base_url.host == "openrouter.ai"
    finally:
        provider.close()


def test_deepseek_flash_from_openrouter_is_an_allowed_radar_model():
    settings = replace(RadarSettings(), model="deepseek/deepseek-v4.1-flash",
                       research_model="deepseek/deepseek-v4.1-flash", openrouter_api_key="test-key")
    assert settings.configuration_error() is None


def test_search_capacity_settings_accept_the_documented_env_maxima(monkeypatch):
    monkeypatch.setenv("RADAR_SEARCH_BRANCHES", "8")
    monkeypatch.setenv("RADAR_MAX_SOURCES", "128")
    monkeypatch.setenv("RADAR_MAX_CANDIDATES", "64")
    settings = load_radar_settings()
    assert (settings.branches, settings.max_sources, settings.max_candidates) == (8, 128, 64)


def test_remote_deadline_is_capped_at_provider_limit(monkeypatch):
    monkeypatch.setenv("RADAR_PROVIDER", "openrouter")
    monkeypatch.setenv("RADAR_DEADLINE_SECONDS", "1600")
    assert load_radar_settings().deadline_seconds == 1200


def test_cloud_profile_targets_eighteen_minutes(monkeypatch):
    monkeypatch.setenv("RADAR_PROVIDER", "openrouter")
    monkeypatch.setenv("RADAR_DEADLINE_SECONDS", "1080")
    assert load_radar_settings().deadline_seconds == 1080


def test_local_provider_can_use_a_longer_deadline(monkeypatch):
    monkeypatch.setenv("RADAR_PROVIDER", "local")
    monkeypatch.setenv("RADAR_DEADLINE_SECONDS", "1600")
    assert load_radar_settings().deadline_seconds == 1600


def test_audio_transcription_is_disabled_without_url_and_allowlist_is_explicit(monkeypatch):
    monkeypatch.delenv("RADAR_AUDIO_TRANSCRIPTION_URL", raising=False)
    monkeypatch.delenv("RADAR_AUDIO_TRANSCRIPTION_ALLOWLIST", raising=False)
    settings = load_radar_settings()
    assert settings.audio_transcription_url == ""
    assert settings.audio_transcription_allowlist == ()

    monkeypatch.setenv("RADAR_AUDIO_TRANSCRIPTION_URL", "http://whisper:9000/v1/audio/transcriptions")
    monkeypatch.setenv("RADAR_AUDIO_TRANSCRIPTION_ALLOWLIST", "media.example,cdn.example")
    settings = load_radar_settings()
    assert settings.audio_transcription_url.endswith("/v1/audio/transcriptions")
    assert settings.audio_transcription_allowlist == ("media.example", "cdn.example")
