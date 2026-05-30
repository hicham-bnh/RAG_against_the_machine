from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Any


class Answer:
    def __init__(self) -> None:
        """
        instance of the model and the necessary methods
        """
        model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def answer(self, qst: str, sources: Any, chunks: List) -> str:
        """
        generating the answer to the question with the LLM
        """
        extraits = "\n\n".join([chunk.content[:600] for chunk in chunks[:4]])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer the question using ONLY the provided excerpts. "
                    "Be concise and direct. "
                    "Do not add any information not present in the excerpts."
                )
            },
            {
                "role": "user",
                "content": f"Excerpts:\n{extraits}\n\nQuestion: {qst}"
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = outputs[0][input_len:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        if isinstance(response, list):
            response = " ".join(response)
        return response.strip()
