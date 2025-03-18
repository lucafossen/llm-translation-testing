from models.interfaces import TraditionalMTInterface

class Google(TraditionalMTInterface):
    name = "google"

    def __init__(self):
        from google.cloud import translate_v2 as translate
        super().__init__()
        self.client = translate.Client()

    def translate(self, text, source, target):
        convert = {'eng_Latn': 'en', 'mri_Latn':'mi', 'nob_Latn':'no'}
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        result = self.client.translate(text, source_language=convert[source], target_language=convert[target])
        return result["translatedText"]