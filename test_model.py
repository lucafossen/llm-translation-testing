# Import necessary classes and functions
from prompt import n_shot_translate_prompt
from models.huggingface_models import Mistral7B as Model
from main import TranslationTask

# Define source and target languages
source_language = 'eng_Latn'
target_language = 'nob_Latn'

# Create an instance of the model
model = Model()

# Create a translation task
translation_task = TranslationTask(source_language, target_language, model)

# Define a source text
source_text = "We now have 4-month-old mice that are non-diabetic that used to be diabetic."

# Generate a translation prompt
prompt = n_shot_translate_prompt(source_text, source_language, target_language, n=0)

# Print the generated prompt
print(prompt)

# Get the translation result
result = model.get_result(prompt)

# Print the translation result
print(result)
