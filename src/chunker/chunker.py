from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


class Chunker:
    def __init__(self) -> None:
        pass

    def chunk_data(self):
        directory = Path("data/raw/vllm-0.10.1")
        files = [file for file in directory.rglob('*') if file.is_file()]
