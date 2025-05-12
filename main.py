from models.interfaces import *
from models.openai_models import *
from models.huggingface_models import *
from models.replicate_models import *
from models.traditional_mt import *
from models.thebloke_quants import *
from visualization import Visualization
import os
from datetime import datetime
import os
from translation_task import TranslationTask, TranslationTaskManager
import json

# TODO


def create_tasks(config, selected_models=None, selected_pairs=None):
    # Create a list of tasks to run
    tasks: list[TranslationTask] = []

    # Create all possible language pairs
    pairs = []
    for lang1 in config['langs']:
        for lang2 in config['langs']:
            if lang1 != lang2:
                pairs.append((lang1, lang2))

    # If selected_pairs is not None, filter pairs
    # (this allows for finer control over which pairs to run)
    if selected_pairs is not None:
        pairs = [pair for pair in pairs if pair in selected_pairs]

    # Create a task for each model and language pair
    for model in config['models']:
        # If selected_models is not None, skip models not in selected_models
        if selected_models is not None and model not in selected_models:
            continue
        for pair in pairs:
            task = TranslationTask(pair[0], pair[1], model)
            tasks.append(task)

    print("Created tasks")
    return tasks


def main(config):
    """
    Main function.
    """
    # Create tasks (one task represents a full translation run through a dataset)
    tasks: list[TranslationTask] = create_tasks(config)

    # Task manager helps with both running and evaluating tasks
    task_manager = TranslationTaskManager()

    # Define stop sequence (kind of a hack, but it works)
    # Currently only the evaluation module uses this, running tasks will not guarantee to stop at this sequence (Because of wonky Mistral behaviour)
    LLMInterface.stop = config['stop_sequence']

    # Run tasks (loads existing results if they exist)
    if config['run_tasks']:
        task_manager.run_multiple_tasks(tasks, threading='off')
    # If not running, just load existing results
    else:
        for task in tasks:
            loaded = task.load_existing_results()
            if loaded:
                print(f"Loaded {task.result_file}")
            else:
                print(f"Could not load {task.result_file}")

    if config['run_evaluation']:
        for task in tasks:
            task.calculate_scores()
            print(task.source_language_name, "->", task.target_language_name, task.llm.name)
            print(f'Average BLEU score: {task.bleu_score:.4f}')
            print("\n")

        # Calculate and print averages for BLEU
        individual_scores_bleu, scores_by_model_bleu, scores_by_pair_bleu, scores_by_language_bleu, scores_by_language_model_bleu = task_manager.get_averages(tasks, metric="bleu", print_results=True)

        # Calculate and print averages for chrF
        individual_scores_chrf, scores_by_model_chrf, scores_by_pair_chrf, scores_by_language_chrf, scores_by_language_model_chrf = task_manager.get_averages(tasks, metric="chrf", print_results=True)

        # Get current time for output directory
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Plot the results for BLEU
        output_dir = os.path.join(config['viz_output_dir'], current_time, "bleu")
        os.makedirs(output_dir, exist_ok=True)
        viz_bleu = Visualization(output_location=output_dir, metric="BLEU")
        viz_bleu.provide_data(individual_scores_bleu, scores_by_model_bleu, scores_by_pair_bleu, scores_by_language_bleu, scores_by_language_model_bleu)
        viz_bleu.plot_all()

        # Plot the results for chrF
        output_dir = os.path.join(config['viz_output_dir'], current_time, "chrf")
        os.makedirs(output_dir, exist_ok=True)
        viz_chrf = Visualization(output_location=output_dir, metric="chrF")
        viz_chrf.provide_data(individual_scores_chrf, scores_by_model_chrf, scores_by_pair_chrf, scores_by_language_chrf, scores_by_language_model_chrf)
        viz_chrf.plot_all()

        # Write accuracy report to a text file
        report_file = os.path.join(config['viz_output_dir'], current_time, "scores_report.json")
        report_data = {
            "bleu": individual_scores_bleu,
            "chrf": individual_scores_chrf,
        }
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=4)
        print(f"Individual scores report written to {report_file}")
    if config['run_bootstrap']:
        # Run bootstrap analysis
        for task in tasks:
            task.run_bootstrap_analysis()

# Define config
# (this is the only place where you need to change the code to run different models)
# The config is a dictionary with two keys:
# - models: a list of LLMs to run
# - langs: a list of languages to run
# The code will run all combinations of models and languages.
if __name__ == "__main__":
    config = {
                'run_tasks': True,
                'run_evaluation': True,
                'viz_output_dir': 'output_graphs',
                'run_bootstrap': False, # Not implemented yet
                'models':
                    [
                    TheBlokeLlama2_13B_Q2_K_GGUF,
                    TheBlokeLlama2_13B_Q3_K_S_GGUF,
                    TheBlokeLlama2_13B_Q3_K_M_GGUF,
                    TheBlokeLlama2_13B_Q3_K_L_GGUF,
                    TheBlokeLlama2_13B_Q4_0_GGUF,
                    TheBlokeLlama2_13B_Q4_K_S_GGUF,
                    TheBlokeLlama2_13B_Q4_K_M_GGUF,
                    TheBlokeLlama2_13B_Q5_0_GGUF,
                    TheBlokeLlama2_13B_Q5_K_S_GGUF,
                    TheBlokeLlama2_13B_Q5_K_M_GGUF,
                    TheBlokeLlama2_13B_Q6_K_GGUF,
                    TheBlokeLlama2_13B_Q8_0_GGUF
                    ],

                'langs':
                    ['eng_Latn', 'nob_Latn'], # English, Maori

                'stop_sequence': '\n'
            }
    main(config)