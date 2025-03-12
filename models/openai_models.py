from models.interfaces import LLMInterface
from openai import OpenAI

class OpenAiInterface(LLMInterface):
    def __init__(self):
        self.client = OpenAI()
        self.model = None
        self.temperature = 0.7

    def get_result(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.create_messages_from_prompt(prompt),
            max_tokens=500,
            temperature=self.temperature,
            stop=self.stop
        )
        result = response.choices[0].message.content.strip()
        return result

    def create_messages_from_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        return messages

class GPT_3_5(OpenAiInterface):
    name = "GPT-3.5"

    def __init__(self):
        super().__init__()
        self.model = "gpt-3.5-turbo"

class GPT_4(OpenAiInterface):
    name = "GPT-4"

    def __init__(self):
        super().__init__()
        self.model = "gpt-4"