from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Mapping

from ..core.models import Hit


class RetrievalBackend(ABC):
    """Abstract retrieval backend to be evaluated.

    Backends are responsible for indexing documents and serving search
    requests. The evaluator is agnostic to how this is implemented.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name (for logging and display)."""

    @abstractmethod
    def index_corpus(self, corpus: Mapping[str, Mapping[str, str]]) -> None:
        """Index a BEIR-style corpus.

        `corpus` is a mapping from doc_id to a dict with at least
        `"title"` and `"text"` keys.
        """

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Hit]:
        """Execute a search query and return ranked hits."""

    def bulk_search(self, queries: Mapping[str, str], top_k: int) -> Dict[str, List[Hit]]:
        """Optional optimized multi-query search.

        Default implementation calls `search` per query.
        """

        return {qid: self.search(text, top_k=top_k) for qid, text in queries.items()}
