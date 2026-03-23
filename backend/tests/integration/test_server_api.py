from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from smartfiles.server.api import app, state


def test_health_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "vector_store_initialized" in data


class DummyEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        # Return a single 3-dimensional embedding per text
        return [[0.1, 0.2, 0.3] for _ in texts]


class DummyVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[list[float]], int]] = []

    def search(self, query_embeddings: list[list[float]], k: int = 5) -> list[dict[str, Any]]:
        self.calls.append((query_embeddings, k))
        return [
            {
                "id": "dummy-1",
                "text": "dummy result",
                "score": 99.0,
                "filepath": "/dummy/path.pdf",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 1,
            }
        ]


def test_search_empty_query_returns_empty_list() -> None:
    client = TestClient(app)
    response = client.get("/search", params={"query": "   "})
    assert response.status_code == 200
    assert response.json() == []


def test_search_uses_embedder_and_vector_store(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    dummy_embedder = DummyEmbedder()
    dummy_store = DummyVectorStore()

    # Bypass real startup initialization by setting state directly
    state.embedder = dummy_embedder
    state.vector_store = dummy_store

    client = TestClient(app)
    response = client.get("/search", params={"query": "test", "k": 3})

    assert response.status_code == 200
    data = response.json()

    # Ensure our dummy vector store response came back
    assert isinstance(data, list)
    assert len(data) == 1
    item = data[0]
    assert item["id"] == "dummy-1"
    assert item["text"] == "dummy result"
    assert item["score"] == 99.0

    # Ensure embedder and store were called
    assert dummy_embedder.calls
    assert dummy_store.calls
    embeddings, k = dummy_store.calls[0]
    assert len(embeddings) == 1
    assert k == 3
