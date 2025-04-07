from models.interfaces import LLMInterface
from ctransformers import AutoModelForCausalLM
from abc import ABC, abstractmethod

class TheBlokeLlama2_13B_GGUF(LLMInterface, ABC):
    def __init__(
        self,
        model_file,
        cls="TheBloke/Llama-2-13B-GGUF",
        model_type="llama",
        gpu_layers=50
        ):

        self.llm = AutoModelForCausalLM.from_pretrained(cls, model_file=model_file, model_type=model_type, gpu_layers=gpu_layers)

    def get_result(self, prompt) -> str:
        return self.llm(prompt)

class TheBlokeLlama2_13B_Q2_K_GGUF(TheBlokeLlama2_13B_GGUF):
    def __init__(self, model_file="llama-2-13b.Q2_K.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_Q3_K_S_GGUF(TheBlokeLlama2_13B_GGUF):
    def __init__(self, model_file="llama-2-13b.Q3_K_S.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_Q3_K_M_GGUF(TheBlokeLlama2_13B_GGUF):
    def __init__(self, model_file="llama-2-13b.Q3_K_M.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_Q3_K_L_GGUF(TheBlokeLlama2_13B_GGUF):
    def __init__(self, model_file="llama-2-13b.Q3_K_L.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_Q4_0_GGUF(TheBlokeLlama2_13B_GGUF):
    def __init__(self, model_file="llama-2-13b.Q4_0.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_Q4_K_S_GGUF(TheBlokeLlama2_13B_GGUF):
    def __init__(self, model_file="llama-2-13b.Q4_K_S.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_Q4_K_M_GGUF(TheBlokeLlama2_13B_GGUF):
    def __init__(self, model_file="llama-2-13b.Q4_K_M.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_Q5_0_GGUF(TheBlokeLlama2_13B_GGUF):
    def __init__(self, model_file="llama-2-13b.Q5_0.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_Q5_K_S_GGUF(TheBlokeLlama2_13B_GGUF):
    def __init__(self, model_file="llama-2-13b.Q5_K_S.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_Q5_K_M_GGUF(TheBlokeLlama2_13B_GGUF):
    def __init__(self, model_file="llama-2-13b.Q5_K_M.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_Q6_K_GGUF(TheBlokeLlama2_13B_GGUF):
    def __init__(self, model_file="llama-2-13b.Q6_K.gguf"):
        super().__init__(model_file=model_file)

class TheBlokeLlama2_13B_Q8_0_GGUF(TheBlokeLlama2_13B_GGUF):
    def __init__(self, model_file="llama-2-13b.Q8_0.gguf"):
        super().__init__(model_file=model_file)