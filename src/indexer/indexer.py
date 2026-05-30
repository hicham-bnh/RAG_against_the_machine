import bm25s
import json
from src.validation.validation import Chunk
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

    def load(self):
        index = bm25s.BM25.load("data/processed/bm25_index")
        with open("data/processed/chunks/chunks.json", "r") as fd:
            data = json.load(fd)
        chunks = [Chunk(** d) for d in data]
        return index, chunks
