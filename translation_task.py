from evaluation import EvaluationModule
from iso639 import Lang
import itertools
import json
import threading
from tqdm import tqdm
from prompt import n_shot_translate_prompt
import threading
import json
import os
from models.interfaces import LLMInterface, TraditionalMTInterface

class TranslationTask:
    """
    One translation task represents a specific combination of a language pair and a model.
    Running through a translation task means translating a dataset from one language to another.
    """

    evaluation_module = EvaluationModule()

    def __init__(self, source_language: str, target_language: str, llm: LLMInterface):
        self.source_texts = []  # Store true source text
        self.prompts = []  # Store prompts
        self.true_target_texts = []  # Store true target text
        self.pred_target_texts = [] # Store prediction results

        # Store language pair and model
        self.source_language = source_language
        self.target_language = target_language
        self.source_language_name = Lang(self.source_language[0:3]).name
        self.target_language_name = Lang(self.target_language[0:3]).name
        self.llm = llm
        # Store BLEU and chrF scores (only calculated when needed with self.calculate_scores()
        self.bleu_score = None
        self.chrf_score = None

        # Create a lock for thread safety
        self.lock = threading.Lock()

        # Store target (ground truth) lines as an instance variable.
        # This will be saved alongside predictions, and makes for easier comparison later on.
        with open(f"flores200_dataset/devtest/{self.target_language}.devtest", 'r', encoding='utf-8') as f:
            self.source_lines = f.readlines()

        # Set result file path
        self.result_file = f"results/{self.source_language[0:3]}_{self.target_language[0:3]}_{self.llm.name.replace('/', '_')}.result.json"
        
        # Check if this task has already been completed
        self.completed = self.check_if_completed()
        if self.completed:
            self.load_existing_results()

    def check_if_completed(self):
        """
        Checks if the translation task has already been completed.
        Returns True if completed, False otherwise.
        """

        if os.path.exists(self.result_file):
            print(f"Result file {self.result_file} exists")
            with open(self.result_file, 'r', encoding='utf-8') as f:
                result_dict = json.load(f)
                return len(result_dict["translations"]) == len(self.source_lines)
        else:
            print(f"Result file {self.result_file} does not exist.")
            return False


    def handle_completed_task(self, pbar=None):
        """
        Handles the case when the task has already been completed.
        Returns True if the task is completed and should be skipped, False otherwise.
        """
        if self.completed:
            print(f"Task {self.result_file} has been fully completed. Skipping...")
            # Update progress bar
            if pbar is not None:
                with self.lock:
                    pbar.update(len(self.source_lines))
            return True
        return False

    def calculate_scores(self):
        """
        Calculate BLEU and chrF scores for this task, making them available through self.get_bleu_score() and self.get_chrf_score().
        """
        # Normalize and tokenize the sentences
        self.bleu_score, self.chrf_score = self.evaluation_module.calculate_scores(self)

    def get_bleu_score(self):
        if self.bleu_score is None:
            self.calculate_scores()
        return self.bleu_score

    def get_chrf_score(self):
        if self.chrf_score is None:
            self.calculate_scores()
        return self.chrf_score
    
    def run_bootstrap(self, interval=0.95):
        """
        Run bootstrap analysis for this task.
        """
        pass

    def load_existing_results(self):
        """
        Loads existing results from file.
        Returns True if results were loaded, False otherwise.
        """
        if os.path.exists(self.result_file):
            print(f"Loading existing results from {self.result_file}...", end="")
            with open(self.result_file, 'r', encoding='utf-8') as f:
                # Load results from file
                result_dict = json.load(f)
                # Store results in python lists (we will continue appending to these lists)
                self.source_language = result_dict["source_lang"]
                self.target_language = result_dict["target_lang"]
                self.prompts = [t["prompt"] for t in result_dict["translations"]]
                self.source_texts = [t["source_text"] for t in result_dict["translations"]]
                # Stop sequence is applied, so we split on it and take the first part
                self.pred_target_texts = [t["pred_target_text"].split(LLMInterface.stop)[0] for t in result_dict["translations"]]
                self.true_target_texts = [t["true_target_text"].split(LLMInterface.stop)[0] for t in result_dict["translations"]]
            print(" Loaded results.")
            return True
        else:
            return False

    def get_start_index(self):
        """
        Returns the index of the first line that has not been translated yet.
        """
        if os.path.exists(self.result_file):
            with open(self.result_file, 'r', encoding='utf-8') as f:
                result_dict = json.load(f)
                return len(result_dict["translations"])
        return 0

    def run_task(self, start=0, end=-1, pbar=None):
        """
        Run a translation task for a specific language pair and model.
        Please note that the line numbers are 0-indexed, so start=0 refers to the first line in the file.
        """
        # Check if this task has already been completed
        if self.handle_completed_task(pbar):
            return

        # Load results if they exist
        if self.load_existing_results() == True:
            # Update start index
            start = self.get_start_index()
            # Update progress bar
            if pbar is not None:
                with self.lock:
                    pbar.update(start)
            print(f" Loaded {start} results. {len(self.source_lines) - start} to go.")

        # Read source lines from file
        with open(f"flores200_dataset/devtest/{self.source_language}.devtest", 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Translates the whole file by default
        if end == -1:
            end = len(lines)

        num_translations = 0  # Number of translations done so far
        for line in lines[start:end]:

            # Store source and true target text
            self.source_texts.append(line.strip())
            self.true_target_texts.append(self.source_lines[start+num_translations].strip())

            if issubclass(type(self.llm), TraditionalMTInterface):
                # Translate with Google
                translated_text = self.llm.translate(line, self.source_language, self.target_language)
                self.prompts.append(None)
            elif issubclass(type(self.llm), LLMInterface):
                # Create prompt
                prompt = n_shot_translate_prompt(line, self.source_language, self.target_language, n=0)
                self.prompts.append(prompt)

                # Translate prompt
                translated_text = self.llm.get_result(prompt)
            print("\n" + "*"*50 + translated_text)
            self.pred_target_texts.append(translated_text)

            # Update progress bar
            if pbar is not None:
                with self.lock:
                    pbar.update()

            # Save results every 10 translations
            if (num_translations + 1) % 10 == 0:
                self.save_translation_result()
            num_translations += 1
        print(f"\nFinished translation {self.source_language_name} -> {self.target_language_name} ({num_translations}/{end-start})")
        self.save_translation_result()

    def save_translation_result(self):
        translations = []
        for i in range(len(self.prompts)):
            translations.append({
                "prompt": self.prompts[i],
                "source_text": self.source_texts[i],
                "pred_target_text": self.pred_target_texts[i],
                "true_target_text": self.true_target_texts[i],
                "index": i
            })

        result_dict = {
            "source_lang": self.source_language,
            "target_lang": self.target_language,
            "translations": translations
        }

        with open(self.result_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)



class TranslationTaskManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.pbar = None

    def run_multiple_tasks(self, tasks: list[TranslationTask], threading: str = "by_task"):
        # Create progress bar
        total_translations = sum(len(task.source_lines) for task in tasks)
        self.pbar = tqdm(total=total_translations, desc="Running tasks")

        if threading == "by_model":
            self.run_tasks_by_model(tasks)
        elif threading == "by_task":
            self.run_tasks_separately(tasks)
        elif threading == "off":
            self.run_unthreaded_tasks(tasks)

    def run_tasks_separately(self, tasks: list[TranslationTask]):
        """
        Run tasks separately, i.e. one thread per task. This is the default.
        This is faster, but may incur rate limiting if you run too many tasks at once (haven't seen this yet).
        """
        threads = []

        for task in tasks:
            thread = threading.Thread(target=self.run_task, args=(task, self.pbar))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        self.pbar.close()

    def run_tasks_by_model(self, tasks: list[TranslationTask]):
        """
        Run tasks by model, i.e. one thread per model.
        This is slower, but safer since it shouldn't run the risk of hitting rate limiting.
        """
        # Group tasks by model
        tasks_by_model = {}
        for task in tasks:
            if task.llm.name not in tasks_by_model:
                tasks_by_model[task.llm.name] = []
            tasks_by_model[task.llm.name].append(task)

        threads = []

        # Create a thread for each model
        for model_tasks in tasks_by_model.values():
            thread = threading.Thread(target=self.run_multiple_tasks, args=(model_tasks, self.pbar))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        self.pbar.close()

    def run_unthreaded_tasks(self, tasks: list[TranslationTask]):
        """
        Sort the tasks by model, instantiate each model, run tasks for that model, then delete the model instance.
        This is by far the slowest option, but is suitable for running local models when you have limited memory.

        This is a slightly hacky approach since the original task system includes a llm attribute,
        but here we are extracting it and instantiating it separately.
        """
        # Group tasks by model
        tasks_by_model = {}
        for task in tasks:
            if task.llm not in tasks_by_model:
                tasks_by_model[task.llm] = []
            tasks_by_model[task.llm].append(task)

        print(f"tasks_by_model: {tasks_by_model}")

        # Iterate over each model groups
        for model, model_tasks in tasks_by_model.items():
            print(f"Running tasks unthreaded for model: {model.name}")

            # If the task is already completed, skip it so we avoid loading the model
            skip = True
            for task in model_tasks:
                if not task.handle_completed_task(pbar=self.pbar):
                    skip = False
            if skip:
                continue

            # Instantiate the model
            model_class = model_tasks[0].llm
            model_instance = model_class()
            print(f"PRINTING DETAILS 4: {model.name}")
            try:
                # Hugging-Face model / pipeline
                print(model_instance.generation_config)      # shows sampling defaults
                print(model_instance.config)                 # shows architectural hyper-params
            except:
                try:
                    # CTransformers (GGUF) model
                    print(model_instance.config)                   # shows sampling defaults
                    print("n_ctx =", model_instance.context_length)
                except:
                    print("Error occurred while printing model details")

            # Run ONLY the tasks that still need work
            pending_tasks = [t for t in model_tasks if not t.completed]

            # Assign the new model to each task and run tasks sequentially
            for t in pending_tasks:
                t.llm = model_instance
                t.run_task(pbar=self.pbar)
                t.llm = None # De-reference the model instance (to free up memory later)

            print(f"Deleting model: {model.name}")
            # Delete (de-reference) the model instance, this shuold bring the reference count to 0
            del model_instance
            print(f"Deleted model: {model.name}")

    def run_task(self, task, pbar):
        """
        Run a single task.
        Might be redundant (consider removing and refactoring).
        """
        task.run_task(pbar=pbar)


    # def run_multiple_tasks(self, tasks, pbar):
    #     """
    #     Run multiple tasks.
    #     """
    #     for task in tasks:
    #         task.run_task(pbar=pbar)
    #     # with self.lock:
    #     #     pbar.update()

    # Inside the TranslationTaskManager class, add these methods
    def get_averages(self, tasks, metric="bleu", print_results=False):
        individual_scores = {}
        scores_by_model = {}
        scores_by_pair = {}
        scores_by_language = {}
        scores_by_language_model = {}  # New dictionary for language scores broken down by model

        for task in tasks:
            model_name = task.llm.name
            pair = (task.source_language, task.target_language)

            if metric == "bleu":
                metric_score = task.get_bleu_score()
            elif metric == "chrf":
                metric_score = task.get_chrf_score()

            # Language pair and model (This shouldn't be here since it's not an average; I think it is a good idea to move al of this functionality into evaluation.py)
            individual_scores.setdefault(task.source_language, {}).setdefault(task.target_language, {}).setdefault(model_name, []).append(metric_score)

            # Average per model
            scores_by_model.setdefault(model_name, []).append(metric_score)

            # Average per language pair
            scores_by_pair.setdefault(pair, []).append(metric_score)

            # Average per language
            scores_by_language.setdefault(task.source_language, []).append(metric_score)
            scores_by_language.setdefault(task.target_language, []).append(metric_score)

            # Average per language broken down by model
            if task.source_language not in scores_by_language_model:
                scores_by_language_model[task.source_language] = {}
            if task.target_language not in scores_by_language_model:
                scores_by_language_model[task.target_language] = {}

            scores_by_language_model[task.source_language].setdefault(model_name, []).append(metric_score)
            scores_by_language_model[task.target_language].setdefault(model_name, []).append(metric_score)

        if print_results:
            for model, scores in scores_by_model.items():
                print(f"Model {model}: Average {metric} score: {sum(scores) / len(scores):.4f}")

            for pair, scores in scores_by_pair.items():
                print(f"Pair {pair}: Average {metric} score: {sum(scores) / len(scores):.4f}")

            for lang, scores in scores_by_language.items():
                print(f"Language {lang}: Average {metric} score: {sum(scores) / len(scores):.4f}")

            for lang, scores in scores_by_language_model.items():
                print(f"Language {lang}:")
                for model, model_scores in scores.items():
                    print(f"\tModel {model}: Average {metric} score: {sum(model_scores) / len(model_scores):.4f}")

        return individual_scores, scores_by_model, scores_by_pair, scores_by_language, scores_by_language_model