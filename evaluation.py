import spacy
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf
from tqdm import tqdm

# Load the SpaCy multi-language model to use for tokenization
nlp_model = "xx_ent_wiki_sm"
try:
    nlp = spacy.load(nlp_model, disable=["parser", "ner"])
except:
    print(f"Downloading the SpaCy model '{nlp_model}' for tokenization...")
    spacy.cli.download(nlp_model)
    nlp = spacy.load(nlp_model, disable=["parser", "ner"])

class EvaluationModule:
    def calculate_scores(self, task, use_tokenizer=True, identical_references=True):
        # TODO: handle identical references flag
        """
        Calculate the average BLEU and chrF scores for the given task, with an option to use simple split.

        Args:
            task: An instance of a translation task containing true and predicted target texts.
            use_tokenize: A boolean flag to determine whether to use SpaCy tokenization or simple whitespace split.

        Returns:
            A tuple containing the average BLEU score and the average chrF score as floats.
        """
        # Using SpaCy tokenizer is recommended, but it is slower than simple split
        if use_tokenizer:
            # Tokenize the reference and candidate sentences using SpaCy with tqdm progress bar
            print(f"Tokenizing for {task.source_language} to {task.target_language} with {task.llm.name} translation...")
            tokenized_references = [[[str(token) for token in nlp(ref.lower())]] for ref in tqdm(task.true_target_texts, desc="Tokenizing references")]
            tokenized_candidates = [[str(token) for token in nlp(candidate.lower())] for candidate in tqdm(task.pred_target_texts, desc="Tokenizing candidates")]
        else:
            # Tokenize the reference and candidate sentences using .split() with tqdm progress bar
            tokenized_references = [[ref.lower().split()] for ref in tqdm(task.true_target_texts, desc="Splitting references")]
            tokenized_candidates = [candidate.lower().split() for candidate in tqdm(task.pred_target_texts, desc="Splitting candidates")]

        # Flatten the list of references for each source sentence for chrF
        flat_references = [[ref for refs in refs_group for ref in refs] for refs_group in tokenized_references]

        # Calculate the BLEU score using the tokenized sentences
        avg_bleu_score = corpus_bleu(tokenized_references, tokenized_candidates)

        # Calculate the corpus-level chrF score using the tokenized sentences
        avg_chrf_score = corpus_chrf(flat_references, tokenized_candidates)

        return avg_bleu_score, avg_chrf_score

# Usage within the same file, tests the module
if __name__ == "__main__":
    from main import TranslationTask
    from models.openai_models import GPT_4

    # Get a model
    nlp_model = GPT_4()
    # Create a task
    task = TranslationTask("eng_Latn", "nob_Latn", nlp_model)
    task.load_existing_results()

    # Create an evaluation module
    evaluation_module = EvaluationModule()
    # Evaluate the task
    avg_bleu_score, avg_chrf_score = evaluation_module.calculate_scores(task, use_tokenizer=True)
    print("BLEU score:", avg_bleu_score)
    print("chrF score:", avg_chrf_score)