from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1"


@dataclass
class EmbeddingModel:
    model: SentenceTransformer

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings as lists of floats."""
        if not texts:
            return []
        vectors = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


def get_default_embedding_model() -> EmbeddingModel:
    model = SentenceTransformer(DEFAULT_MODEL_NAME)
    return EmbeddingModel(model=model)
