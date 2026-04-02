from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_KEY = "all-minilm-l6-v2"
MODEL_ENV_VAR = "SMARTFILES_EMBEDDING_MODEL"
PROFILE_ENV_VAR = "SMARTFILES_EMBEDDING_PROFILE"


@dataclass(frozen=True)
class SupportedEmbeddingModel:
    key: str
    model_id: str
    description: str


SUPPORTED_MODELS: Dict[str, SupportedEmbeddingModel] = {
    "all-minilm-l6-v2": SupportedEmbeddingModel(
        key="all-minilm-l6-v2",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        description="Default; very small, fast 384d model suitable for local/dev.",
    ),
    "bge-small-en-v1": SupportedEmbeddingModel(
        key="bge-small-en-v1",
        model_id="BAAI/bge-small-en-v1.5",
        description=(
            "General-purpose English model (768d), good trade-off of speed and quality. "
            "Mapped to the current Hugging Face repo 'BAAI/bge-small-en-v1.5'."
        ),
    ),
    "bge-base-en-v1": SupportedEmbeddingModel(
        key="bge-base-en-v1",
        model_id="BAAI/bge-base-en-v1.5",
        description=(
            "Larger English model (1024d) for higher-quality embeddings. "
            "Mapped to the current Hugging Face repo 'BAAI/bge-base-en-v1.5'."
        ),
    ),
}


@dataclass
class EmbeddingModel:
    model: SentenceTransformer

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings as lists of floats."""
        if not texts:
            return []
        vectors = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


def _resolve_model_id() -> str:
    """Resolve which embedding model ID/path to load.

    Precedence (highest to lowest):

    1. `SMARTFILES_EMBEDDING_MODEL` – explicit model id or local path.
    2. `SMARTFILES_EMBEDDING_PROFILE` – key in SUPPORTED_MODELS.
    3. Default profile `bge-small-en-v1`.
    """

    direct = os.getenv(MODEL_ENV_VAR)
    if direct:
        return direct

    profile = os.getenv(PROFILE_ENV_VAR, DEFAULT_MODEL_KEY)
    cfg = SUPPORTED_MODELS.get(profile)
    if cfg is not None:
        return cfg.model_id

    # Fallback: treat the profile value as a raw model id/path.
    return profile


def get_default_embedding_model() -> EmbeddingModel:
    """Return the default embedding model.

    By default this uses the `all-minilm-l6-v2` profile, which
    corresponds to the Hugging Face model
    `sentence-transformers/all-MiniLM-L6-v2`.

    You can override this in two ways:

    - Set `SMARTFILES_EMBEDDING_PROFILE` to one of the known keys in
      `SUPPORTED_MODELS` (e.g. `bge-base-en-v1`, `all-minilm-l6-v2`).
    - Set `SMARTFILES_EMBEDDING_MODEL` to either a Hugging Face model
      id or a local filesystem path to a SentenceTransformers model
      directory. This takes precedence over the profile.

    This keeps the call site abstract (only `embed_texts` is used) and
    makes it easy to experiment with different local open-source
    models by swapping env vars, without changing application code.
    """

    model_id = _resolve_model_id()
    model = SentenceTransformer(model_id)
    return EmbeddingModel(model=model)


def list_supported_models() -> List[SupportedEmbeddingModel]:
    """Return the list of built-in supported embedding model profiles."""

    return list(SUPPORTED_MODELS.values())
