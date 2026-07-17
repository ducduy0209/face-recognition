import pytest

from app.config import Settings, load_settings


def test_load_settings_reads_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("MATCH_THRESHOLD", "0.5")
    s = load_settings()
    assert s.api_key == "secret"
    assert s.match_threshold == 0.5
    assert s.model_name == "buffalo_l"
    assert s.min_face_size == 80


def test_load_settings_requires_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        load_settings()
