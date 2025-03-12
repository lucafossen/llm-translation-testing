from models.interfaces import LLMInterface
import replicate

class Replicate(LLMInterface):
    def __init__(self):
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
        result = "".join(output)
        return result

class Llama_2_70b_chat(Replicate):
    name = "llama-2-70b-chat"

    def __init__(self):
        super().__init__()
        self.model = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"

class Llama_2_70b_base(Replicate):
    name = "llama-2-70b-base"

    def __init__(self):
        super().__init__()
        self.model = "meta/llama-2-70b:a52e56fee2269a78c9279800ec88898cecb6c8f1df22a6483132bea266648f00"

class Mistral_7b_instruct(Replicate):
    name = "mistral-7b-instruct"

    def __init__(self):
        super().__init__()
        self.model = "mistralai/mistral-7b-instruct-v0.1:83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70"

class Mistral_7b_base(Replicate):
    name = "mistral-7b-base"

    def __init__(self):
        super().__init__()
        self.model = "mistralai/mistral-7b-v0.1:3e8a0fb6d7812ce30701ba597e5080689bef8a013e5c6a724fafb108cc2426a0"

class Vicuna_13b(Replicate):
    name = "vicuna-13b"

    def __init__(self):
        super().__init__()
        self.model = "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b"