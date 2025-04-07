# Import necessary classes and functions
from prompt import n_shot_translate_prompt
from models.openai_models import GPT_4 as Model
from main import TranslationTask

# Define source and target languages
source_language = 'eng_Latn'
target_language = 'nob_Latn'

# Create an instance of the model
model = Model()

# Create a translation task
translation_task = TranslationTask(source_language, target_language, model)

# Define a source text
source_text = "The axioms do not make any claims about anything so long as the terms they use have no definitions, and therefore, no meaning other than the one that axioms give them."

# Generate a translation prompt
prompt = n_shot_translate_prompt(source_text, source_language, target_language, n=0)

# Print the generated prompt
print(prompt)

# Get the translation result
result = model.get_result(prompt)[len(prompt):]

# Print the translation result
print(result)
