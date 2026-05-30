import json
from src.validation.validation import StudentSearchResults
from src.validation.validation import RagDataset, AnsweredQuestion
from typing import Any


class Evaluation:
    def evaluate(self, student_answer_path: str, dataset_path: str) -> None:
        """
        Evaluate the student's search results using Recall@k metrics.
        """
        with open(student_answer_path) as f:
            data = json.load(f)
        student_results = StudentSearchResults(**data)
        with open(dataset_path) as f:
            data = json.load(f)
        dataset = RagDataset(**data)
        ground_truth = {}
        for question in dataset.rag_questions:
            if isinstance(question, AnsweredQuestion):
                ground_truth[question.question_id] = question.sources
        print("Evaluation Results")
        print("========================================")
        print(f"Questions evaluated: {len(dataset.rag_questions)}")
        for k in [1, 3, 5, 10]:
            recalls = []
            for result in student_results.search_results:
                correct_sources = ground_truth.get(result.question_id, [])
                if not correct_sources:
                    continue
                retrieved_sources = result.retrieved_sources[:k]
                found_sources = 0
                for correct_source in correct_sources:
                    for retrieved_source in retrieved_sources:
                        if self.overlap_ratio(
                            correct_source, retrieved_source
                        ) >= 0.05:
                            found_sources += 1
                            break
                recalls.append(found_sources / len(correct_sources))
            print(f"Recall@{k}: {sum(recalls)/len(recalls):.3f}")

    def overlap_ratio(self, source_a: Any, source_b: Any) -> float:
        """
            Calculate the overlap ratio between two text
            sources based on character indices.
        """
        first = max(
            source_a.first_character_index,
            source_b.first_character_index
        )
        last = min(
            source_a.last_character_index,
            source_b.last_character_index
        )
        overlap = max(0, last - first)
        size = source_a.last_character_index - source_a.first_character_index
        return overlap / size if size > 0 else 0.0
