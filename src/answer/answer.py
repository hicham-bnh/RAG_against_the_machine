from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Answer:
    def __init__(self) -> None:
        model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def answer(self, qst, sources, chunks):
        extraits = "\n\n".join([chunk.content for chunk in chunks])
        prompt = f"""You are an expert Python code assistant.
        answer the question based only on the provided excerpts.
        Your answer must be concise, precise and in English.
        Excerpts:
        {extraits}
        Question: {qst}
        Answer:"""
        token_prompt = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**token_prompt, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return
