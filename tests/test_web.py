import pytest

import app as web_app


@pytest.fixture
def client():
    web_app.app.config["TESTING"] = True
    yield web_app.app.test_client()
    web_app.app.config["TESTING"] = False


def test_home_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"GraphTopic" in response.data


@pytest.mark.parametrize(
    "form",
    [
        {"window": "not-a-number"},
        {"window": "41"},
        {"resolution": "nan"},
        {"resolution": "5.1"},
        {"topn": "4"},
    ],
)
def test_web_analysis_rejects_invalid_parameters(client, form):
    response = client.post("/analyze", data={"source": "paste", **form})

    assert response.status_code == 400
    assert any(
        message in response.data
        for message in (b"Window", b"Resolution", b"Top words", b"valid numbers")
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"documents": "not a list"},
        {"documents": [None] * 20},
        {"documents": ["text"] * 19},
        {"documents": ["text"] * 12_001},
        {"documents": ["text"] * 20, "window": 41},
        {"documents": ["text"] * 20, "resolution": float("inf")},
    ],
)
def test_api_rejects_invalid_analysis_requests(client, monkeypatch, payload):
    def unexpected_fit(*_args, **_kwargs):
        pytest.fail("invalid input must be rejected before fitting")

    monkeypatch.setattr(web_app, "fit_model", unexpected_fit)
    response = client.post("/api/analyze", json=payload)

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_api_analysis_success_response_shape(client, monkeypatch):
    class FakeModel:
        communities = [{1, 2}]
        topics_ = [{"id": 0, "words": [("galaxy", 0.5), ("orbit", 0.3)]}]

        @staticmethod
        def summary():
            return {"n_topics": 1}

    def fake_fit(texts, window, resolution, topn):
        assert len(texts) == 20
        assert (window, resolution, topn) == (8, 1.0, 10)
        return FakeModel()

    monkeypatch.setattr(web_app, "fit_model", fake_fit)
    response = client.post("/api/analyze", json={"documents": ["space text"] * 20})

    assert response.status_code == 200
    assert response.get_json() == {
        "summary": {"n_topics": 1},
        "topics": [{"id": 0, "words": ["galaxy", "orbit"]}],
    }


def test_inference_rejects_expired_session(client):
    response = client.post("/infer/not-a-session", json={"text": "space"})

    assert response.status_code == 404
    assert "error" in response.get_json()
