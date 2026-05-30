import fire
import json
from tqdm import tqdm
from pathlib import Path
from src.validation.validation import RagDataset, MinimalSearchResults, StudentSearchResults
from src.validation.validation import StudentSearchResultsAndAnswer, MinimalAnswer
from src.answer.answer import Answer
from src.chunker.chunker import Chunker
from src.indexer.indexer import Indexer
from src.search.search import Search

class Student:
    def index(self, max_chunk_size=2000):
        chunker = Chunker()
        chunk = chunker.chunk_data("data/raw/vllm-0.10.1", max_chunk_size)
        indexer = Indexer()
        indexer.indexing(chunk)
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, qst, k=10):
        indexer = Indexer()
        retriever, chunks = indexer.load()
        sercher = Search()
        result = sercher.search(qst, k, chunks, retriever)
        print(result)

    def search_dataset(self, dataset_path, k=10, save_directory="data/output/search_results") -> None:
        sercher = Search()
        with open(dataset_path, "r") as fd:
            data = json.load(fd)
        data_result = RagDataset(** data)
        indexer = Indexer()
        retriever, chunks = indexer.load()
        serach_result = []
        for question in tqdm(data_result.rag_questions, desc="Searching..."):
            sources = sercher.search(question.question, k, chunks, retriever)
            serach_result.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources
            ))
        result = StudentSearchResults(
            search_results=serach_result,
            k=k
        )
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        file_name = Path(dataset_path).name
        output_path = Path(save_directory) / file_name
        with open(output_path, "w") as f:
            json.dump(result.model_dump(), f)

    def answer(self, qst, k=10):
        answer = Answer()
        indexer = Indexer()
        retriever, chunks = indexer.load()
        sercher = Search()
        soucres = sercher.search(qst, k, chunks, retriever)
        chunks_dict = {(c.file_path, c.first_character_index): c for c in chunks}
        source_chunks = [chunks_dict[(s.file_path, s.first_character_index)] for s in soucres]
        result = answer.answer(qst, soucres, source_chunks)
        print(result)

    def answer_dataset(self, dataset_path, save_directory="data/output/search_results", k=10):
        file_output = []
        answer = Answer()
        sercher = Search()
        indexer = Indexer()
        retriever, chunks = indexer.load()
        with open(dataset_path, "r") as fd:
            qsts = json.load(fd)
        data_result = RagDataset(**qsts)
        for question in tqdm(data_result.rag_questions, desc="Answering..."):
            soucres = sercher.search(question.question, k, chunks, retriever)
            chunks_dict = {(c.file_path, c.first_character_index): c for c in chunks}
            source_chunks = [chunks_dict[(s.file_path, s.first_character_index)] for s in soucres]
            result = answer.answer(question.question, soucres, source_chunks)
            file_output.append(MinimalAnswer(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=soucres,
                answer=result
            ))
        result_final = StudentSearchResultsAndAnswer(
            search_results=file_output,
            k=k
        )
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        file_name = Path(dataset_path).name
        output_path = Path(save_directory) / file_name
        with open(output_path, "w") as f:
            json.dump(result_final.model_dump(), f)

    def evaluate(self, student_answer_path, dataset_path, k=10):
        from src.evaluation.evaluation import Evaluation
        evaluator = Evaluation()
        evaluator.evaluate(student_answer_path, dataset_path)

if __name__ == "__main__":
    fire.Fire(Student)