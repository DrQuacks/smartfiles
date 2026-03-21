from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1"
MODEL_ENV_VAR = "SMARTFILES_EMBEDDING_MODEL"


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
        """Return the default embedding model.

        By default this loads the public Hugging Face model
        `BAAI/bge-small-en-v1`. You can override this by setting the
        `SMARTFILES_EMBEDDING_MODEL` environment variable to either:

        - Another model id on the Hugging Face hub, or
        - A local filesystem path to a compatible SentenceTransformers
            model directory.

        This makes it easy to work fully offline after the initial
        download: download the model once, point the env var at the local
        copy, and SmartFiles will never hit the network for embeddings.
        """

        model_name = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL_NAME)
        model = SentenceTransformer(model_name)
        return EmbeddingModel(model=model)
