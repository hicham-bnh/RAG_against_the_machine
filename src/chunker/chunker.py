from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import PythonCodeTextSplitter
from pathlib import Path
from typing import List
from src.validation.validation import Chunk


class Chunker:
    def __init__(self) -> None:
        pass

    def chunk_data(self,path: str, chunk_size: int = 2000) -> List:
        result = []
        directory = Path(path)
        files = [file for file in directory.rglob('*') if file.is_file()]
        for file in files:
            if file.suffix == ".py":
                result.extend(self.chunk_code(str(file), chunk_size))
            elif file.suffix == ".md":
                result.extend(self.chunk_doc(str(file), chunk_size))
        return result

    def chunk_code(self, file: str, size: int) -> List:
        with open(file, 'r', encoding="utf-8", errors="ignore") as fd:
            data = fd.read()
        result = []
        splitter = PythonCodeTextSplitter(
            chunk_size=size,
            chunk_overlap=0
        )
        chunks = splitter.create_documents([data])
        position = 0
        for chunk in chunks:
            first = data.find(chunk.page_content, position)
            last = first + len(chunk.page_content)
            result.append(Chunk(
                file_path=file,
                content=chunk.page_content,
                first_character_index=first,
                last_character_index=last
            ))
            position = last
        return result

    def chunk_doc(self, file: str, size: int) -> List:
        with open(file, 'r', encoding="utf-8", errors="ignore") as fd:
            data = fd.read()
        result = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=0
        )
        chunks = splitter.create_documents([data])
        position = 0
        for chunk in chunks:
            first = data.find(chunk.page_content, position)
            last = first + len(chunk.page_content)
            result.append(Chunk(
                file_path=file,
                content=chunk.page_content,
                first_character_index=first,
                last_character_index=last
            ))
            position = last
        return result
