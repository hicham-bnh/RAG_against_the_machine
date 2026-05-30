import bm25s
from src.indexer.indexer import Indexer
from src.validation.validation import MinimalSource


class Search:
    def __init__(self) -> None:
        pass

    def search(self, qst: str, k: int, chunks, retrieve):
        result = []
        qst_token = bm25s.tokenize([qst])
        results, scores = retrieve.retrieve(qst_token, k=k)
        for chunk_idx in results[0]:
            chunk = chunks[chunk_idx]
            result.append(MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index
            ))
        return result