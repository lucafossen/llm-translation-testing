from models.interfaces import LLMInterface

class HuggingfaceLLM(LLMInterface):
    def __init__(self, model_name, max_length=200):
        from transformers import pipeline
        self.name = model_name
        self.pipeline = pipeline('text-generation', model=model_name)
        self.max_length = max_length

    def get_result(self, prompt):
        response = self.pipeline(prompt, max_length=self.max_length)
        result = response[0]['generated_text'].strip()
        return result

class Mistral7B(HuggingfaceLLM):
    name = "huggingface mistral-7B-v0.1"

    def __init__(self):
        super().__init__('mistralai/Mistral-7B-v0.1')