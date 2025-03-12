from abc import ABC, abstractmethod

class LLMInterface(ABC):
    """
    This is the Language Learning Model (LLM) interface. It provides a common interface for different language models.
    Each model should implement the 'get_result' method, which takes a prompt and returns the LLM result.
    """
    name = None
    stop = "\n"  # Stop sequence for the LLM (when to stop generating text)

    @abstractmethod
    def get_result(self, prompt) -> str:
        raise NotImplementedError

class TraditionalMTInterface(ABC):
    """
    This is the Traditional machine translation (MT) interface. It provides a common interface for different traditional translation methods.
    Each method should implement the 'translate' method, which takes a string, a source and target language, and returns the translation result.
    """
    name = None

    @abstractmethod
    def translate(self, text, source, target) -> str:
        raise NotImplementedError