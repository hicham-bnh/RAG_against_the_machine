import fire
import json
from tqdm import tqdm
from pathlib import Path
from src.validation.validation import RagDataset, MinimalSearchResults
from src.validation.validation import StudentSearchResults
from src.validation.validation import StudentSearchResultsAndAnswer
from src.validation.validation import MinimalAnswer
from src.answer.answer import Answer
from src.chunker.chunker import Chunker
from src.indexer.indexer import Indexer
from src.search.search import Search


class Student:
    def index(self, max_chunk_size: int = 2000) -> None:
        """
        indexing chunks and saving them in the data/processed
        """
        chunker = Chunker()
        chunk = chunker.chunk_data("data/raw/vllm-0.10.1", max_chunk_size)
        indexer = Indexer()
        indexer.indexing(chunk)
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, qst: str, k: int = 10) -> None:
        """
        search for the most coherent k chunks for the answer
        """
        indexer = Indexer()
        retriever, chunks = indexer.load()
        sercher = Search()
        result = sercher.search(qst, k, chunks, retriever)
        print(result)

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 10,
            save_directory: str = "data/output/search_results"
    ) -> None:
        """
        search for the most coherent k chunks for the list of answer
        """
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
                question_str=question.question,
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
            json.dump(result.model_dump(), f, indent=2)
        print(f"Saved student_search_results to {output_path}")

    def answer(self, qst: str, k: int = 10) -> None:
        """
        generating the answer to the question with the LLM
        """
        answer = Answer()
        indexer = Indexer()
        retriever, chunks = indexer.load()
        sercher = Search()
        soucres = sercher.search(qst, k, chunks, retriever)
        chunks_dict = {
            (c.file_path, c.first_character_index): c for c in chunks
        }
        source_chunks = [
            chunks_dict[
                (s.file_path, s.first_character_index)
            ] for s in soucres
        ]
        result = answer.answer(qst, soucres, source_chunks)
        print(result)

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str = "data/output/search_results_and_answer",
            k: int = 10
    ) -> None:
        """
        generating the answer to the list of uestion with the LLM
        """
        file_output = []
        answer = Answer()
        indexer = Indexer()
        retriever, chunks = indexer.load()
        with open(student_search_results_path, "r") as fd:
            data = json.load(fd)
        search_results = StudentSearchResults(**data)
        chunks_dict = {
            (c.file_path, c.first_character_index): c for c in chunks
        }
        print(f"Loaded {len(search_results.search_results)} ", end=" ")
        print(f"questions from {student_search_results_path}")
        for item in tqdm(search_results.search_results, desc="Answering..."):
            sources = item.retrieved_sources
            source_chunks = [
                chunks_dict[(s.file_path, s.first_character_index)]
                for s in sources
                if (s.file_path, s.first_character_index) in chunks_dict
            ]
            result = answer.answer(item.question_str, sources, source_chunks)
            file_output.append(MinimalAnswer(
                question_id=item.question_id,
                question_str=item.question_str,
                retrieved_sources=sources,
                answer=str(result)
            ))
        print(f"Processed {len(file_output)} of", end=" ")
        print(f"{len(search_results.search_results)} questions")
        result_final = StudentSearchResultsAndAnswer(
            search_results=file_output,
            k=search_results.k
        )
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        file_name = Path(student_search_results_path).name
        output_path = Path(save_directory) / file_name
        with open(output_path, "w") as f:
            json.dump(result_final.model_dump(), f, indent=2)
        print(f"Saved student_search_results_and_answer to {output_path}")

    def evaluate(
            self,
            student_answer_path: str,
            dataset_path: str,
            k: int = 10
    ) -> None:
        """
        Evaluate the student's search results using Recall@k metrics.
        """
        from src.evaluation.evaluation import Evaluation
        evaluator = Evaluation()
        evaluator.evaluate(student_answer_path, dataset_path)


if __name__ == "__main__":
    try:
        fire.Fire(Student)
    except Exception as e:
        print(e)
