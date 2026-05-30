from src.chunker.chunker import Chunker
from src.search.search import Search
from src.indexer.indexer import Indexer

test = Chunker()
indxer = Indexer()
retriever, chunks = indxer.load()
indxer.indexing(test.chunk_data())
search = Search()
result = search.search("how to add two function?", 10, retriever, chunks)
