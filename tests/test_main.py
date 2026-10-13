from fastapi.testclient import TestClient

from app.main import MAX_TEXT_LENGTH, app

client = TestClient(app)


def test_health_reports_configured_model() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "tinyllama"}


def test_homepage_serves_summarizer() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Shortform" in response.text


def test_summarize_rejects_blank_text() -> None:
    response = client.post("/api/summarize", json={"text": "  \n  "})

    assert response.status_code == 422
    assert "non-whitespace" in response.json()["detail"][0]["msg"]


def test_summarize_rejects_text_over_limit() -> None:
    response = client.post("/api/summarize", json={"text": "a" * (MAX_TEXT_LENGTH + 1)})

    assert response.status_code == 422
