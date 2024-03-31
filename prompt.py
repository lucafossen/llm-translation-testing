from iso639 import Lang

def return_n_lines(filename, n):
    with open(filename, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    return lines[:n]

def n_shot_translate_prompt(source_text, source_langcode, target_langcode, n):
    """
    Returns a prompt for the n-shot translation task.
    """
    lines_source = return_n_lines(f"flores200_dataset/dev/{source_langcode}.dev", n)
    lines_target = return_n_lines(f"flores200_dataset/dev/{target_langcode}.dev", n)
    source_name = Lang(source_langcode[0:3]).name
    target_name = Lang(target_langcode[0:3]).name
    prompt = f"Translate the following {n} sentences from {source_name} to {target_name}:\n"
    for i in range(n):
        prompt += f"Source ({source_name}): {lines_source[i]}"
        prompt += f"Translation ({target_name}): {lines_target[i]}"
        prompt += "\n"
    
    prompt += f"Source ({source_name}): {source_text}"
    prompt += f"Translation ({target_name}): "
    return prompt