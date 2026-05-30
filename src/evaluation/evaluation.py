import json
from pathlib import Path
from src.validation.validation import StudentSearchResults, RagDataset, AnsweredQuestion


class Evaluation:
    def evaluate(self, student_answer_path, dataset_path):
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
        for k in [1, 3, 5, 10]:
            recalls = []
            for result in student_results.search_results:
                bonnes = ground_truth.get(result.question_id, [])
                if not bonnes:
                    continue
                tes_sources = result.retrieved_sources[:k]
                trouvees = 0
                for bonne in bonnes:
                    for ta_source in tes_sources:
                        if self.overlap_ratio(bonne, ta_source) >= 0.05:
                            trouvees += 1
                            break
                recalls.append(trouvees / len(bonnes))
            print(f"Recall@{k}: {sum(recalls)/len(recalls):.3f}")

    def overlap_ratio(self, source_a, source_b) -> float:
        first = max(source_a.first_character_index, source_b.first_character_index)
        last = min(source_a.last_character_index, source_b.last_character_index)
        chevauchement = max(0, last - first)
        taille = source_a.last_character_index - source_a.first_character_index
        return chevauchement / taille if taille > 0 else 0.0
