# LLM Translation Testing

A growing test harness for numerically evaluating machine translation performance on a range of large language models, with a focus on English and Māori translation.

## Overview

This project provides a framework to:
- Run translation tasks using various Large Language Models (LLMs), including different quantized versions.
- Evaluate the translation quality using BLEU and chrF metrics.
- Visualize the performance of different models.
- Support for translation between any supported language in the FLORES-200 dataset.
- Facilitate manual comparison of translation outputs.

The primary dataset used is FLORES-200, specifically the devtest sets.

## Features

- **Multiple Model Support**: Test various LLMs, including Hugging Face (HF) and GGUF quantized models (e.g., LLaMA 2 13B Chat and its variants).
- **Automated Evaluation**: Calculates BLEU and chrF scores for translation outputs.
- **Data Visualization**: Generates graphs to compare model performance, saved in the `output_graphs/` directory.
- **Result Persistence**: Saves raw translation results in JSON format in the `results/` directory.
- **Configuration-driven**: Experiments are configured through a Python dictionary in the main notebook.
- **Colab Integration**: Supports running in Google Colab, including GPU acceleration.
- **Exploratory Data Analysis**: Includes an `eda.ipynb` notebook for dataset analysis.
- **Manual Evaluation Support**: Scripts and reports for manual review of translations (see `manual_evaluation/` and `mini-data-analysis/generate_comparison_report.py`).

## Directory Structure

```
.
├── models/                # Potentially for storing or defining models
├── results/               # Stores raw JSON results from translation tasks
├── output_graphs/         # Stores generated graphs and visualizations
├── manual_evaluation/     # Contains reports and data for manual translation comparison
│   └── `manual_evaluation/comparison_report_mri_eng.md`
├── mini-data-analysis/    # Small, ad-hoc analysis scripts
│   └── `mini-data-analysis/generate_comparison_report.py`
│   └── `README.md`
├── `main.ipynb`             # Main Jupyter notebook for running experiments and evaluation
├── `main.py`                # Core Python script with main logic, imported by `main.ipynb`
├── `evaluation.py`          # Script for evaluating translation quality
├── `translation_task.py`    # Defines and manages translation tasks
├── `visualization.py`       # Script for generating result visualizations
├── `prompt.py`              # Handles prompt engineering for LLMs
├── `eda.ipynb`              # Notebook for Exploratory Data Analysis of the dataset
├── `test_model.ipynb`       # Notebook for testing individual models
├── `test_model.py`          # Script for testing individual models
├── `requirements.txt`       # Python dependencies
├── `README.md`              # This file
└── ...                    # Other configuration and project files
```

## Setup

### Prerequisites
- Python 3 (tested with Python 3.11, see `main.ipynb`)
- Pip (Python package installer)
- Git (for cloning, if applicable)
- NVIDIA GPU and CUDA (recommended for GGUF models with `ctransformers[cuda]`)

### Dependencies
Install the required Python packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```
Alternatively, if running in an environment like Google Colab, the `main.ipynb` notebook includes cells to install specific packages:
```python
# From main.ipynb
# !pip install iso639-lang
# !pip install replicate
# !pip install ctransformers[cuda]>=0.2.24
```

### Dataset
The project uses the Flores200 dataset. Ensure the `flores200_dataset` directory is populated with the necessary language files (e.g., `eng.dev`, `mri.dev`, `eng.devtest`, `mri.devtest`). The `flores200_dataset.zip` file in the repository might contain this dataset.

### Google Colab Setup
If using Google Colab, the initial cells in `main.ipynb` handle mounting Google Drive and changing the working directory:
```python
# filepath: main.ipynb
# ...existing code...
try:
    # Comment out if not using colab
    from google.colab import drive
    drive.mount('/content/drive')

    # Specific for luca's computer
    %cd "/content/drive/Othercomputers/lucas-yoga/Work/Te Taka & Albert LLM project/llm-translation-testing"
    using_colab = True
except:
    print("Not using Google Colab")
    using_colab = False
# ...existing code...
```
Adjust the `%cd` path as per your Google Drive structure.

## Usage

The main entry point for running experiments is the `main.ipynb` notebook.

1.  **Open and Configure**: Open `main.ipynb` in a Jupyter environment or Google Colab.
2.  **Set Configuration**: Modify the `config` dictionary in the notebook to define which models to run, languages to translate between, and other parameters.
    ```python
    # filepath: main.ipynb
    # ...existing code...
    config = {
                'run_tasks': False, # Set to True to run new translation tasks
                'run_evaluation': True, # Set to True to evaluate existing/new results
                'viz_output_dir': 'output_graphs',
                'run_bootstrap': False, # Not implemented yet
                'models':
                    [
                    # Example models (refer to main.ipynb for the full list)
                    LLaMA2_13B_Chat_HF,
                    TheBlokeLlama2_13B_chat_Q2_K_GGUF,
                    # ... other models
                    ],

                'langs':
                    ['eng_Latn', 'mri_Latn'], # English, Maori

                'stop_sequence': '
'
            }
    # ...existing code...
    ```
    - `run_tasks`: Set to `True` to execute translations. If `False`, it might load existing results if available.
    - `run_evaluation`: Set to `True` to perform BLEU score evaluation.
    - `models`: A list of model identifiers/objects to be used for translation. These are typically defined in `main.py` or imported.
    - `langs`: A list of language codes (e.g., `eng_Latn` for English, `mri_Latn` for Māori).

3.  **Run Main Function**: Execute the cell calling the `main` function with the configuration:
    ```python
    # filepath: main.ipynb
    # ...existing code...
    main(config)
    # ...existing code...
    ```

4.  **View Results**:
    -   Translation outputs are saved as JSON files in the `results` directory (e.g., `results/eng_mri_LLaMA 2 13B Chat - HF.result.json`).
    -   Evaluation scores (like average BLEU scores) are typically printed in the notebook output.
    -   Visualization graphs are saved in the `output_graphs` directory. An individual scores report will also be generated (e.g., `output_graphs/YYYY-MM-DD_HH-MM-SS/scores_report.json`).

## Evaluation

-   **Automated**: The project uses BLEU and chrF scores for quantitative evaluation, as implemented in `evaluation.py`. Results are shown in the `main.ipynb` output.
-   **Manual**: For qualitative analysis, translations can be manually compared. The script `mini-data-analysis/generate_comparison_report.py` can be used to generate markdown reports like `manual_evaluation/comparison_report_mri_eng.md` for specific sentences across different models.

## Key Files

-   `main.ipynb`: Orchestrates the translation and evaluation tasks.
-   `main.py`: Contains the core logic, model definitions, and the `main` function.
-   `evaluation.py`: Implements the BLEU score calculation and other evaluation metrics.
-   `translation_task.py`: Manages the process of translating texts with the specified models.
-   `visualization.py`: Generates plots and visual representations of the evaluation results.
-   `prompt.py`: Likely contains functions or classes for formatting prompts fed to the LLMs.
-   `eda.ipynb`: Notebook for performing exploratory data analysis on the Flores200 dataset.
