from models.interfaces import LLMInterface
from ctransformers import AutoModelForCausalLM
from abc import ABC, abstractmethod

class TheBlokeLlama2_13B_chat_GGUF(LLMInterface):
    def __init__(
        self,
        model_file,
        cls="TheBloke/Llama-2-13B-chat-GGUF",
        model_type="llama",
        gpu_layers=50
        ):

        self.llm = AutoModelForCausalLM.from_pretrained(cls, model_file=model_file, model_type=model_type, gpu_layers=gpu_layers)

    def get_result(self, prompt) -> str:
        return self.llm(prompt, max_new_tokens=1000)

class TheBlokeLlama2_13B_chat_Q2_K_GGUF(TheBlokeLlama2_13B_chat_GGUF):
    name = "TheBloke Llama2 13B Chat - Q2_K (GGUF)"
    def __init__(self, model_file="llama-2-13b-chat.Q2_K.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_chat_Q3_K_S_GGUF(TheBlokeLlama2_13B_chat_GGUF):
    name = "TheBloke Llama2 13B Chat - Q3_K_S (GGUF)"
    def __init__(self, model_file="llama-2-13b-chat.Q3_K_S.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_chat_Q3_K_M_GGUF(TheBlokeLlama2_13B_chat_GGUF):
    name = "TheBloke Llama2 13B Chat - Q3_K_M (GGUF)"
    def __init__(self, model_file="llama-2-13b-chat.Q3_K_M.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_chat_Q3_K_L_GGUF(TheBlokeLlama2_13B_chat_GGUF):
    name = "TheBloke Llama2 13B Chat - Q3_K_L (GGUF)"
    def __init__(self, model_file="llama-2-13b-chat.Q3_K_L.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_chat_Q4_0_GGUF(TheBlokeLlama2_13B_chat_GGUF):
    name = "TheBloke Llama2 13B Chat - Q4_0 (GGUF)"
    def __init__(self, model_file="llama-2-13b-chat.Q4_0.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_chat_Q4_K_S_GGUF(TheBlokeLlama2_13B_chat_GGUF):
    name = "TheBloke Llama2 13B Chat - Q4_K_S (GGUF)"
    def __init__(self, model_file="llama-2-13b-chat.Q4_K_S.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_chat_Q4_K_M_GGUF(TheBlokeLlama2_13B_chat_GGUF):
    name = "TheBloke Llama2 13B Chat - Q4_K_M (GGUF)"
    def __init__(self, model_file="llama-2-13b-chat.Q4_K_M.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_chat_Q5_0_GGUF(TheBlokeLlama2_13B_chat_GGUF):
    name = "TheBloke Llama2 13B Chat - Q5_0 (GGUF)"
    def __init__(self, model_file="llama-2-13b-chat.Q5_0.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_chat_Q5_K_S_GGUF(TheBlokeLlama2_13B_chat_GGUF):
    name = "TheBloke Llama2 13B Chat - Q5_K_S (GGUF)"
    def __init__(self, model_file="llama-2-13b-chat.Q5_K_S.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_chat_Q5_K_M_GGUF(TheBlokeLlama2_13B_chat_GGUF):
    name = "TheBloke Llama2 13B Chat - Q5_K_M (GGUF)"
    def __init__(self, model_file="llama-2-13b-chat.Q5_K_M.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_chat_Q6_K_GGUF(TheBlokeLlama2_13B_chat_GGUF):
    name = "TheBloke Llama2 13B Chat - Q6_K (GGUF)"
    def __init__(self, model_file="llama-2-13b-chat.Q6_K.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_chat_Q8_0_GGUF(TheBlokeLlama2_13B_chat_GGUF):
    name = "TheBloke Llama2 13B Chat - Q8_0 (GGUF)"
    def __init__(self, model_file="llama-2-13b-chat.Q8_0.gguf"):
        super().__init__(model_file=model_file)