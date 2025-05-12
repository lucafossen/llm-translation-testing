import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class Visualization:
    def __init__(self, output_location='.', metric="None"):
        self.individual_scores = {}
        self.scores_by_model = {}
        self.scores_by_pair = {}
        self.scores_by_language = {}
        self.scores_by_language_model = {}
        self.output_location = output_location
        self.metric = metric

    def provide_data(self, individual_scores, scores_by_model, scores_by_pair, scores_by_language, scores_by_language_model):
        # Calculate averages
        self.individual_scores = individual_scores
        self.scores_by_model = {model: sum(scores) / len(scores) for model, scores in scores_by_model.items()}
        self.scores_by_pair = {pair: sum(scores) / len(scores) for pair, scores in scores_by_pair.items()}
        self.scores_by_language = {language: sum(scores) / len(scores) for language, scores in scores_by_language.items()}
        self.scores_by_language_model = {
            language: {model: sum(scores) / len(scores) for model, scores in models.items()}
            for language, models in scores_by_language_model.items()
        }

    def save_plot(self, fig, title):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # This line already uses datetime
        filename = f"{self.output_location}/{title.replace(' ', '_').replace('>', '_')}_{timestamp}.png"
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def plot_models(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        models = list(self.scores_by_model.keys())
        scores = list(self.scores_by_model.values())
        ax.bar(models, scores, color='skyblue')
        title = f'Models {self.metric} Scores'
        ax.set_title(title)
        ax.set_ylabel(f'{self.metric} Score')
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        ax.set_xticklabels(models, rotation=45, ha='right')  # Rotate labels to prevent overlap
        self.save_plot(fig, title)

    def plot_language_pairs(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        # Convert the tuple pairs to strings (and remove the suffix for shorter labels)
        pairs = ['>'.join([item.split("_")[0] for item in pair]) for pair in self.scores_by_pair.keys()]
        scores = list(self.scores_by_pair.values())
        ax.bar(pairs, scores, color='lightcoral')
        title = f'Language Pairs {self.metric} Scores (source>target)'
        ax.set_title(title)
        ax.set_ylabel(f'{self.metric} Score')
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        self.save_plot(fig, title)
    
    def plot_all_line_for_language_pairs(self):
        """
        For each language pair in self.scores_by_pair (assumed to be a dictionary mapping
        language pair identifiers to dictionaries of model scores), plot a line plot with
        one point per model and save the plot.
        """
        for language_pair, model_scores in self.scores_by_pair.items():
            # Check that the score for the language pair is a dict of model scores
            if not isinstance(model_scores, dict):
                continue  # Skip if not in the expected format
            fig, ax = plt.subplots(figsize=(8, 4))
            models = sorted(model_scores.keys())
            scores = [model_scores[m] for m in models]
            ax.plot(models, scores, marker='o', linestyle='-', color='blue')
            title = f'Language Pair {language_pair} Model Scores'
            ax.set_title(title)
            ax.set_xlabel('Model')
            ax.set_ylabel(f'{self.metric} Score')
            ax.grid(True, linestyle='--', linewidth=0.5)
            self.save_plot(fig, title)

    def plot_languages(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        languages = list(self.scores_by_language.keys())
        scores = list(self.scores_by_language.values())
        ax.bar(languages, scores, color='lightgreen')
        title = f'Languages {self.metric} Scores'
        ax.set_title(title)
        ax.set_ylabel(f'{self.metric} Score')
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        self.save_plot(fig, title)

    def plot_models_by_language(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.15  # Width of each bar within a group
        languages = list(self.scores_by_language_model.keys())
        language_indices = np.arange(len(languages))  # the label locations
        models = list(self.scores_by_language_model[languages[0]].keys())
        group_width = width * len(models)  # Calculate the total width of each group
        spacing = 0.1  # Space between groups

        for i, model in enumerate(models):
            scores = [self.scores_by_language_model[language][model] for language in languages]
            ax.bar(language_indices + i * width, scores, width, label=model)

        ax.set_ylabel(f'{self.metric} Score')
        ax.set_title(f'Models by Language {self.metric} Scores')
        ax.set_xticks(language_indices + group_width / 2 - width / 2)
        ax.set_xticklabels(languages)
        ax.legend(loc='lower right')
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        self.save_plot(fig, f'Models by Language {self.metric} Scores')

        # Adjust the figure size and layout
        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)

    def plot_languages_by_model(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.2  # the width of the bars
        models = list(next(iter(self.scores_by_language_model.values())).keys())
        model_indices = np.arange(len(models))  # the label locations
        languages = list(self.scores_by_language_model.keys())

        for i, language in enumerate(languages):
            scores = [self.scores_by_language_model[language].get(model, 0) for model in models]
            ax.bar(model_indices + i * width, scores, width, label=language)

        ax.set_ylabel(f'{self.metric} Score')
        ax.set_title(f'Languages by Model {self.metric} Scores')
        ax.set_xticks(model_indices + width / 2 * (len(languages) - 1))
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        self.save_plot(fig, f'Languages by Model {self.metric} Scores')

    def plot_language_pair_by_model(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        language_pairs = ['>'.join(pair) for pair in self.scores_by_pair.keys()]
        models = list(next(iter(self.scores_by_language_model.values())).keys())
        width = 0.15  # Width of each bar within a group
        pair_indices = np.arange(len(language_pairs))  # the label locations
        group_width = width * len(models)  # Calculate the total width of each group
        spacing = 0.1  # Space between groups

        for i, model in enumerate(models):
            scores = [self.scores_by_language_model.get(pair.split('>')[0], {}).get(model, 0) for pair in language_pairs]
            ax.bar(pair_indices + i * width, scores, width, label=model)

        ax.set_ylabel(f'{self.metric} Score')
        ax.set_title(f'Language Pair by Model {self.metric} Scores')
        ax.set_xticks(pair_indices + group_width / 2 - width / 2)
        ax.set_xticklabels(language_pairs)
        ax.legend()
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        self.save_plot(fig, f'Language Pair by Model {self.metric} Scores')

        # Adjust the figure size and layout
        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)

    def plot_all(self):
        self.plot_models()
        self.plot_language_pairs()
        self.plot_all_line_for_language_pairs()
        self.plot_languages()
        self.plot_models_by_language()
        self.plot_languages_by_model()
        self.plot_language_pair_by_model()

if __name__ == "__main__":
    # Example data
    scores_by_model = {'Model A': 0.75, 'Model B': 0.65}
    scores_by_pair = {'en-fr': 0.80, 'en-de': 0.70}
    scores_by_language = {'English': 0.85, 'French': 0.75, 'German': 0.65}
    scores_by_language_model = {
        'English': {'Model A': 0.85, 'Model B': 0.80},
        'French': {'Model A': 0.75, 'Model B': 0.70},
        'German': {'Model A': 0.65, 'Model B': 0.60}
    }

    # Create Visualization instance
    viz = Visualization()

    # Provide data to the Visualization instance
    viz.provide_data(scores_by_model, scores_by_pair, scores_by_language, scores_by_language_model)

    # Plot all the visualizations
    viz.plot_all()