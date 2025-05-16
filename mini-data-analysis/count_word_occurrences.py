import json
import glob
import os
from collections import defaultdict

def count_word_in_results(search_word, results_dir="results", file_prefix="eng_mri"):
    """
    Scans JSON files in the specified directory that start with a given prefix,
    and counts the occurrences of a search_word in 'pred_target_text' and
    'true_target_text' fields.

    Args:
        search_word (str): The word to search for.
        results_dir (str): The directory containing the result files.
        file_prefix (str): The prefix for the JSON files to scan.
    """
    pred_counts = defaultdict(int)
    true_counts = defaultdict(int)
    total_pred_count = 0
    total_true_count = 0
    total_pred_items = 0  # New: Counter for pred_target_text items
    total_true_items = 0  # New: Counter for true_target_text items

    # Convert search word to lowercase for case-insensitive comparison
    search_word_lower = search_word.lower()

    # Construct the path pattern for glob
    path_pattern = os.path.join(results_dir, f"{file_prefix}*.json")
    print(f"Searching in pattern: {path_pattern}")

    for filepath in glob.glob(path_pattern):
        filename = os.path.basename(filepath)
        print(f"Processing file: {filename}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "translations" not in data or not isinstance(data["translations"], list):
                print(f"  Skipping {filename}: 'translations' key not found or not a list.")
                continue

            file_pred_count = 0
            file_true_count = 0
            
            for item_loop_idx, item in enumerate(data["translations"]): # Use enumerate for a loop index
                if not isinstance(item, dict):
                    continue

                # Process pred_target_text
                pred_text = item.get("pred_target_text", "")
                current_item_pred_occurrences = 0
                if pred_text and isinstance(pred_text, str):
                    total_pred_items += 1 # Increment count of pred items
                    current_item_pred_occurrences = pred_text.lower().count(search_word_lower)
                    pred_counts[filename] += current_item_pred_occurrences 
                    total_pred_count += current_item_pred_occurrences    
                    file_pred_count += current_item_pred_occurrences     
                
                # Process true_target_text
                true_text = item.get("true_target_text", "")
                current_item_true_occurrences = 0
                if true_text and isinstance(true_text, str):
                    total_true_items += 1 # Increment count of true items
                    current_item_true_occurrences = true_text.lower().count(search_word_lower)
                    true_counts[filename] += current_item_true_occurrences 
                    total_true_count += current_item_true_occurrences    
                    file_true_count += current_item_true_occurrences     

        except json.JSONDecodeError:
            print(f"  Skipping {filename}: Invalid JSON.")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
        print("-" * 30)

    print("\n--- Summary ---")
    if not glob.glob(path_pattern):
        print(f"No files found matching the pattern: {path_pattern}")
    print(f"Total occurrences of '{search_word}' in 'pred_target_text' across all processed files: {total_pred_count}")
    print(f"Total occurrences of '{search_word}' in 'true_target_text' across all processed files: {total_true_count}")

    # Calculate and print averages
    if total_pred_items > 0:
        average_pred = total_pred_count / total_pred_items
        print(f"Average occurrences of '{search_word}' per 'pred_target_text' item: {average_pred:.2f}")
    else:
        print("No 'pred_target_text' items found to calculate average.")

    if total_true_items > 0:
        average_true = total_true_count / total_true_items
        print(f"Average occurrences of '{search_word}' per 'true_target_text' item: {average_true:.2f}")
    else:
        print("No 'true_target_text' items found to calculate average.")


if __name__ == "__main__":
    # --- Configuration ---
    word_to_search = "whakapapa" # Example word, change as needed
    # Consider making this a command-line argument for more flexibility
    # e.g., using the argparse module:
    # import argparse
    # parser = argparse.ArgumentParser(description="Count word occurrences in JSON translation files.")
    # parser.add_argument("word", help="The word to search for.")
    # parser.add_argument("--dir", default="results", help="Directory containing the result files.")
    # parser.add_argument("--prefix", default="eng_mri", help="Prefix for the JSON files.")
    # args = parser.parse_args()
    # word_to_search = args.word
    # results_folder_name = args.dir
    # file_name_prefix = args.prefix
    # --- End Configuration ---

    results_folder_name = "results" 
    file_name_prefix = "eng_mri"

    # Get the absolute path to the results directory relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_results_dir = os.path.join(script_dir, results_folder_name)
    
    print(f"Searching for word: '{word_to_search}'")
    print(f"Scanning files starting with '{file_name_prefix}' in directory: '{absolute_results_dir}'\n")
    
    count_word_in_results(word_to_search, absolute_results_dir, file_name_prefix)
