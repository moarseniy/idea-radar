import pytest


@pytest.fixture(autouse=True)
def _no_developer_dotenv(monkeypatch):
    """Тесты не видят .env разработчика: ключи и контакты задаются в самом тесте через env."""
    monkeypatch.setattr("app.ml.env._DOTENV", {})
