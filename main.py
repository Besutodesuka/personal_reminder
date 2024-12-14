import json
import csv
import os
import tempfile
from typing import List, Dict

# -----------------------------
# Example: Integration with LLM
# -----------------------------
def query_llm_with_llama_cpp(prompt: str, model_path: str = "path/to/ggml-model.bin") -> str:
    """
    Example function to query a local LLM via llama.cpp using llama-cpp-python.
    
    You need to have llama-cpp-python installed:
    pip install llama-cpp-python
    """
    from llama_cpp import Llama

    llm = Llama(model_path=model_path, n_ctx=2048)
    response = llm(prompt, stop=["Q:", "User:", "LLM:"], echo=False)
    return response["choices"][0]["text"].strip()

def query_llm_with_ollama(prompt: str, model_name: str = "my-model") -> str:
    """
    Example function to query a local LLM via Ollama.
    This assumes you have Ollama installed and running locally.
    """
    import subprocess
    result = subprocess.run(["ollama", "run", model_name], input=prompt.encode('utf-8'), stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').strip()

# Choose which LLM backend to use:
query_llm = query_llm_with_llama_cpp  # or query_llm_with_ollama

# ----------------------------
# Example Data Storage Helpers
# ----------------------------
STORAGE_FILE = "data.csv"

def save_data_to_csv(data: List[Dict[str, str]], file_path: str = STORAGE_FILE):
    """
    Save a list of dictionaries to a CSV file. Assumes keys are consistent.
    """
    if not data:
        return
    
    fieldnames = data[0].keys()
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in data:
            writer.writerow(row)

def load_data_from_csv(file_path: str = STORAGE_FILE) -> List[Dict[str, str]]:
    """
    Load CSV data into a list of dictionaries.
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

# ----------------------------
# Simulating the Interaction
# ----------------------------
def main():
    # Step 1: User tells what happened and what they need
    user_input = "Yesterday I had a meeting about project X from 2pm to 3pm. I also need to schedule a follow-up at 10am tomorrow."
    
    # Step 2: LLM extracts actual activity from user input
    extract_prompt = f"""
    The user provided the following activity details:
    "{user_input}"

    Extract the main activities and their times as structured text without additional commentary.
    Each activity should have a date, title, and detail.
    If a date/time isn't explicitly given, infer a reasonable date or leave it blank.
    """
    extracted_activities = query_llm(extract_prompt)
    
    # Step 3: LLM arranges into JSON
    json_prompt = f"""
    Take the following extracted activities:
    "{extracted_activities}"

    Convert them into a JSON array of objects. 
    Each object should have keys: "date", "title", "detail".
    """
    json_str = query_llm(json_prompt)
    try:
        activities_data = json.loads(json_str)
    except json.JSONDecodeError:
        activities_data = []  # fallback if LLM doesn't return valid JSON

    # Step 4: LLM -> Storage: save current data in csv file
    # We'll trust the data is well-formed. For demonstration, write to CSV.
    save_data_to_csv(activities_data)

    # Step 5: Storage -> LLM: Update memory (Here, "memory" might mean we keep activities_data in memory)
    # In a real scenario, you might store this in a vector DB or memory state. For now, we just keep it in a variable.
    current_memory = activities_data

    # Step 6: User asks for schedule
    user_request = "What is my schedule for tomorrow?"

    # Step 7: LLM ReACT that this command needs data update
    # We'll simulate by prompting the LLM that it should fetch updated data
    react_prompt = f"""
    The user asked: "{user_request}"

    You have some stored activities in a CSV file. 
    Determine if you need to retrieve and update your data from the storage before answering.
    Respond with "NEED_DATA_UPDATE" if you must retrieve data before answering, otherwise "NO_UPDATE_NEEDED".
    """
    react_response = query_llm(react_prompt)

    if "NEED_DATA_UPDATE" in react_response:
        # Step 8: Storage -> LLM: Retrieve Data
        updated_data = load_data_from_csv()
        
        # Step 9: LLM: Augment the data (e.g. summarize, add context)
        augment_prompt = f"""
        You have the following stored activities:
        {json.dumps(updated_data, indent=2)}

        The user wants their schedule for tomorrow. 
        Summarize the activities scheduled for tomorrow in a clear, concise manner.
        If no activities are scheduled, say so.
        """
        schedule_report = query_llm(augment_prompt)
        
        # Step 10: LLM -> User: report what to do
        print("Schedule for tomorrow:")
        print(schedule_report)

    else:
        # If no update needed, just use current_memory
        augment_prompt = f"""
        You have the following activities in memory:
        {json.dumps(current_memory, indent=2)}

        The user wants their schedule for tomorrow. 
        Summarize the activities scheduled for tomorrow in a clear, concise manner.
        If no activities are scheduled, say so.
        """
        schedule_report = query_llm(augment_prompt)
        print("Schedule for tomorrow:")
        print(schedule_report)


if __name__ == "__main__":
    main()
