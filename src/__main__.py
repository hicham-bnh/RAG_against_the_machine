from src.chunker.chunker import Chunker
from src.indexer.indexer import Indexer

test = Chunker()
index = Indexer()
index.indexing(test.chunk_data())
