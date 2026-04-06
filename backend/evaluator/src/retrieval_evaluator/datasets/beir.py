from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from beir.datasets.data_loader import GenericDataLoader

Corpus = Mapping[str, Mapping[str, str]]
Queries = Mapping[str, str]
Qrels = Mapping[str, Mapping[str, int]]


@dataclass
class BeirDataset:
    """Thin wrapper around BEIR's GenericDataLoader.

    `data_dir` should point to the BEIR dataset folder containing
    corpus.jsonl, queries.jsonl, and qrels/.
    """

    name: str
    data_dir: str

    def load(self, split: str = "test") -> Tuple[Corpus, Queries, Qrels]:
        loader = GenericDataLoader(data_folder=self.data_dir)
        corpus, queries, qrels = loader.load(split=split)
        return corpus, queries, qrels
