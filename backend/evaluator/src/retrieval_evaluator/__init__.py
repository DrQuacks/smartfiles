"""Retrieval Evaluator core package.

This package is a prototype for a generic retrieval evaluation harness
that can benchmark arbitrary retrieval backends on BEIR-style datasets.
"""

from .core.models import Hit, RunConfig, RunResult  # noqa: F401
from .backends.base import RetrievalBackend  # noqa: F401
