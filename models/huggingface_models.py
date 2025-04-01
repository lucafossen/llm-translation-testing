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

class LLaMA2_7B_FP16(HuggingfaceLLM):
    name = "LLaMA 2 7B - FP16"

    def __init__(self):
        super().__init__('meta-llama/Llama-2-7b-hf')

class LLaMA2_7B_FP16(HuggingfaceLLM):
    name = "LLaMA 2 7B - FP16"

    def __init__(self):
        super().__init__('meta-llama/Llama-2-7b-hf')

class LLaMA2_7B_8bit(LLMInterface):
    name = "LLaMA 2 7B - 8bit"

    def __init__(self, max_length=200):
        from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        model_name = "meta-llama/Llama-2-7b-hf"
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.name = model_name
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        self.max_length = max_length

    def get_result(self, prompt):
        response = self.pipeline(prompt, max_length=self.max_length)
        result = response[0]['generated_text'].strip()
        return result

class LLaMA2_7B_GPTQ(LLMInterface):
    name = "LLaMA 2 7B - 4bit GPTQ"

    def __init__(self, max_length=200):
        from transformers import AutoTokenizer, pipeline
        from auto_gptq import AutoGPTQForCausalLM

        model_name = "TheBloke/Llama-2-7B-GPTQ"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            device="cuda",
            use_safetensors=True,
            trust_remote_code=True
        )

        self.name = model_name
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        self.max_length = max_length

    def get_result(self, prompt):
        response = self.pipeline(prompt, max_length=self.max_length)
        result = response[0]['generated_text'].strip()
        return result

class LLaMA2_7B_AWQ(LLMInterface):
    name = "LLaMA 2 7B - 4bit AWQ"

    def __init__(self, max_length=200):
        from autoawq import AutoAWQForCausalLM
        from transformers import AutoTokenizer, pipeline

        model_name = "TheBloke/Llama-2-7B-AWQ"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoAWQForCausalLM.from_quantized(model_name, trust_remote_code=True)

        self.name = model_name
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        self.max_length = max_length

    def get_result(self, prompt):
        response = self.pipeline(prompt, max_length=self.max_length)
        result = response[0]['generated_text'].strip()
        return result

class LLaMA2_7B_2bit(LLMInterface):
    name = "LLaMA 2 7B - 2bit (llama.cpp)"

    def __init__(self, max_length=200):
        from llama_cpp import Llama

        self.model_path = "/path/to/llama-2-7b.Q2_K.gguf"  # Download from TheBloke's GGUF repo
        self.llm = Llama(model_path=self.model_path, n_ctx=2048)
        self.max_length = max_length

    def get_result(self, prompt):
        output = self.llm(prompt, max_tokens=self.max_length, stop=["</s>"])
        return output["choices"][0]["text"].strip()

