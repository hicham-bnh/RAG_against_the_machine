import bm25s
import json
from src.validation.validation import Chunk
from pathlib import Path
from typing import List, Tuple, Any


class Indexer:
    def __init__(self) -> None:
        """
        instance of the bm52s module
        """
        self.bms = bm25s.BM25()
        self.token = bm25s.tokenize

    def indexing(self, chunks: List) -> None:
        """
        indexing chunks and saving them in the data/processed
        """
        tokens = self.token([chunk.content for chunk in chunks])
        self.bms.index(tokens)
        self.bms.save("data/processed/bm25_index")
        Path("data/processed/chunks").mkdir(parents=True, exist_ok=True)
        with open("data/processed/chunks/chunks.json", "w") as fd:
            data = [chunk.model_dump() for chunk in chunks]
            json.dump(data, fd, indent=2)

    def load(self) -> Tuple[Any, List[Chunk]]:
        """
        return the save file and the chunks
        """
        index = bm25s.BM25.load("data/processed/bm25_index")
        with open("data/processed/chunks/chunks.json", "r") as fd:
            data = json.load(fd)
        chunks = [Chunk(** d) for d in data]
        return index, chunks
