# In order to save unnecessary overhead, API/model-associated imports are only called when the relevant class is initialized.
# These imports are called in the __init__ method of each class that inherits from LLMInterface or TraditionalMTInterface.
from abc import ABC, abstractmethod

class LLMInterface(ABC):
    """
    This is the Language Learning Model (LLM) interface. It provides a common interface for different language models.
    Each model should implement the 'get_result' method, which takes a prompt and returns the LLM result.
    """
    name = None
    stop = "\n" # Stop sequence for the LLM (when to stop generating text)

    @abstractmethod
    def get_result(self, prompt) -> str:
        raise NotImplementedError

########################################################################################################

class OpenAI(LLMInterface):
    """
    OpenAI is a general class for OpenAI's models, which other specific GPT model classes inherit from.
    """
    def __init__(self):
        # Import OpenAI API
        import openai
        # Initialize LLM specifics
        self.model = None
        self.temperature = 0.7

    def get_result(self, prompt):
        response = openai.ChatCompletion.create(
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

class GPT_3_5(OpenAI):
    """
    Class for the GPT-3.5 model. Inherits from GPT.
    """
    name = "GPT-3.5"

    def __init__(self):
        super().__init__()
        self.model = "gpt-3.5-turbo"

class GPT_4(OpenAI):
    """
    Class for the GPT-4 model. Inherits from GPT.
    """
    name = "GPT-4"

    def __init__(self):
        super().__init__()
        self.model = "gpt-4"

########################################################################################################


class Replicate(LLMInterface):
    """
    Replicate is a general class for the Replicate API, which other specific LLMS inherit from.
    """
    def __init__(self):
        # Import Replicate API
        import replicate
        # Initialize LLM specifics
        self.model = None
        self.temperature = 0.7

    def get_result(self, prompt):
        output = replicate.run(
            self.model,
            input={
                "prompt": prompt,
                "stop_sequences": self.stop,
            }
        )
        result = ""

        for item in output:
            result += item
        return result

class Llama_2_70b_chat(Replicate):
    """
    Class for the Llama 2 70b chat-finetuned model. Inherits from Replicate.
    """
    name = "llama-2-70b-chat"

    def __init__(self):
        super().__init__()
        self.model = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"

class Llama_2_70b_base(Replicate):
    """
    Class for the Llama 2 70b base model. Inherits from Replicate.
    """
    name = "llama-2-70b-base"

    def __init__(self):
        super().__init__()
        self.model = "meta/llama-2-70b:a52e56fee2269a78c9279800ec88898cecb6c8f1df22a6483132bea266648f00"

class Mistral_7b_instruct(Replicate):
    """
    Class for the Mistral 7b instruct-finetuned model. Inherits from Replicate.
    """
    name = "mistral-7b-instruct"

    def __init__(self):
        super().__init__()
        self.model = "mistralai/mistral-7b-instruct-v0.1:83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70"

class Mistral_7b_base(Replicate):
    """
    Class for the Mistral 7b base model. Inherits from Replicate.
    """
    name = "mistral-7b-base"

    def __init__(self):
        super().__init__()
        self.model = "mistralai/mistral-7b-v0.1:3e8a0fb6d7812ce30701ba597e5080689bef8a013e5c6a724fafb108cc2426a0"

class Vicuna_13b(Replicate):
    """
    Class for the Vicuna 13b model. Inherits from Replicate.
    """
    name = "vicuna-13b"

    def __init__(self):
        super().__init__()
        self.model = "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b"

########################################################################################################

class TraditionalMTInterface(ABC):
    """
    This is the Traditional machine translation (MT) interface. It provides a common interface for different traditional translation methods.
    Each method should implement the 'translate' method, which takes a string, a source and target language, and returns the translation result.
    Current issues: langcode compatibility with my setup for LLM models.
    """
    name = None

    @abstractmethod
    def translate(self, text, source, target) -> str:
        raise NotImplementedError

class Google(TraditionalMTInterface):
    """
    Google is a class for interacting with Google Translation API.
    """
    name = "google"

    def __init__(self):
        super().__init__()
        from google.cloud import translate_v2 as translate
        self.client = translate.Client()

    def translate(self, text, source, target):
        # Super hacky way to convert language codes to ISO-639
        convert = {'eng_Latn': 'en', 'mri_Latn':'mi', 'nob_Latn':'no'}
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        result = self.client.translate(text, source_language=convert[source], target_language=convert[target])
        return result["translatedText"]

########################################################################################################

class HuggingfaceLLM(LLMInterface):
    """
    Abstract class for Hugging Face's transformer models. Inherits from LLMInterface.
    This class is not intended to be instantiated directly but to be subclassed by specific model classes.
    """
    def __init__(self, model_name, max_length=200):
        from replicate import pipeline
        self.name = model_name
        self.pipeline = pipeline('text-generation', model=model_name)
        self.max_length = max_length

    def get_result(self, prompt):
        response = self.pipeline(prompt, max_length=self.max_length)
        result = response[0]['generated_text'].strip()
        return result

class Mistral7B(HuggingfaceLLM):
    """
    Class for the Mistral-7B model, inheriting from HuggingfaceLLM.
    """

    name = "huggingface mistral-7B-v0.1"

    def __init__(self):
        super().__init__('mistralai/Mistral-7B-v0.1')