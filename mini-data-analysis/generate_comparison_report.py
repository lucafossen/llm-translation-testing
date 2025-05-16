'''
Generates a markdown report comparing translation outputs for specific indexes from multiple JSON result files.
'''
import json
import os

TARGET_INDEXES = [199, 65, 517, 265, 89, 698]
# Ensure these file names exactly match those in your "results" directory
FILE_NAMES = [
    "mri_eng_LLaMA 2 13B Chat - HF.result.json",
    "mri_eng_TheBloke Llama2 13B Chat - Q4_0 (GGUF).result.json",
    "mri_eng_TheBloke Llama2 13B Chat - Q2_K (GGUF).result.json"
]
RESULTS_DIR = "results" # Relative to the script's location
OUTPUT_FILE = "manual_evaluation/comparison_report_mri_eng.md"

def main():
    '''Main function to generate the comparison report.'''
    # Initialize data structure to hold all extracted information
    # Keyed by index, each value will store source, true target, and a list of predictions
    extracted_data = {
        idx: {
            "source_text": None,
            "true_target_text": None,
            "predictions": [], # List of (file_basename, pred_target_text)
            "found_in_files": set() # To track which files an index was found in
        }
        for idx in TARGET_INDEXES
    }

    # Process each specified file
    for file_name in FILE_NAMES:
        file_path = os.path.join(RESULTS_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}. Skipping.")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}. Skipping.")
            continue

        if "translations" not in data or not isinstance(data["translations"], list):
            print(f"Warning: 'translations' array not found or not a list in {file_path}. Skipping.")
            continue

        # Iterate through translations in the current file
        for item in data["translations"]:
            if not isinstance(item, dict):
                continue # Skip malformed items
            
            item_index = item.get("index")
            if item_index in TARGET_INDEXES:
                # Store source and true target text (assumed consistent across files for the same index)
                # These will be picked from the first file encountered that contains this index.
                if extracted_data[item_index]["source_text"] is None:
                    extracted_data[item_index]["source_text"] = item.get("source_text", "N/A")
                if extracted_data[item_index]["true_target_text"] is None:
                    extracted_data[item_index]["true_target_text"] = item.get("true_target_text", "N/A")
                
                # Store the predicted text for this file
                pred_text = item.get("pred_target_text", "N/A")
                extracted_data[item_index]["predictions"].append((file_name, pred_text))
                extracted_data[item_index]["found_in_files"].add(file_name)

    # Generate Markdown content
    markdown_lines = ["# Translation Comparison Report\n"]

    for index in TARGET_INDEXES:
        markdown_lines.append(f"## Sentence #{index}\n")
        data_for_index = extracted_data[index]

        if data_for_index["source_text"] is None and data_for_index["true_target_text"] is None:
            markdown_lines.append(f"**Note:** Index {index} not found in any of the processed files.\n")
        else:
            source_text = data_for_index['source_text'] if data_for_index['source_text'] else "N/A"
            true_target_text = data_for_index['true_target_text'] if data_for_index['true_target_text'] else "N/A"
            
            markdown_lines.append(f"**Source Text:**\n> {source_text.replace('\n', '\n> ')}\n")
            markdown_lines.append(f"**True Target Text:**\n> {true_target_text.replace('\n', '\n> ')}\n")
            markdown_lines.append("**Predicted Target Texts:**\n")

            # Ensure predictions are grouped by file as they were added
            # And only show predictions from files that actually contained this index
            # The order of files in FILE_NAMES will be preserved for predictions if present.
            predictions_for_current_index = False
            for fn in FILE_NAMES: # Iterate in the defined order of files
                file_predictions = [pred for f, pred in data_for_index["predictions"] if f == fn]
                if file_predictions:
                    predictions_for_current_index = True
                    markdown_lines.append(f"### From `{fn}`:\n")
                    for pred_text in file_predictions:
                        pred_text_formatted = pred_text if pred_text else "N/A"
                        markdown_lines.append(f"> {pred_text_formatted.split('\n')[0]}\n")
            
            if not predictions_for_current_index and (data_for_index["source_text"] is not None):
                 markdown_lines.append("*No predicted texts found for this index in the specified files.*\n")

        markdown_lines.append("\n---\n") # Separator

    # Write to output file
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(markdown_lines))
        print(f"Report generated: {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing output file {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    main()
