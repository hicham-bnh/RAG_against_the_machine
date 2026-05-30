import bm25s
import json
import os
from pathlib import Path


class Indexer:
    def __init__(self) -> None:
        self.bms = bm25s.BM25()
        self.token = bm25s.tokenize

    def indexing(self, chunks):
        tokens = self.token([chunk.content for chunk in chunks])
        self.bms.index(tokens)
        self.bms.save("data/processed/bm25_index")
        Path("data/processed/chunks").mkdir(parents=True, exist_ok=True)
        with open("data/processed/chunks/chunks.json", "w") as fd:
            data = [chunk.model_dump() for chunk in chunks]
            json.dump(data, fd)