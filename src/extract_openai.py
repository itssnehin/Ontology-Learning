# src/extract_openai.py

import os
import json
import openai  # The official OpenAI Python library
from dotenv import load_dotenv

# --- Configuration ---
# This block sets up the environment and paths.

# Load environment variables from the .env file located in the parent 'code/' directory.
# This is a secure way to manage your API key without hardcoding it.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Initialize the OpenAI client.
# The library will automatically find and use the 'OPENAI_API_KEY' from your environment.
try:
    client = openai.OpenAI()
except openai.OpenAIError as e:
    print("Failed to initialize OpenAI client. Is OPENAI_API_KEY set in your .env file?")
    print(f"Error: {e}")
    exit()

# Define the project's directory structure so the script knows where to find files.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_markdown")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "structured_json_openai")


# --- Helper Functions ---
# These small functions handle simple, repeatable tasks.

def load_file(filepath):
    """Loads the entire content of a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return None

def save_json(filepath, data):
    """Saves a Python dictionary to a JSON file with nice formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


# --- Core Extraction Logic ---

def build_extraction_prompt():
    """
    Creates the detailed "system prompt". This instruction tells the LLM its role,
    the task to perform, and the exact format for the output.
    """
    return """
    You are an expert in electronics engineering and knowledge graph creation.
    Your task is to analyze a technical component datasheet, provided in Markdown format,
    and extract key information into a single, structured JSON object.

    The JSON object MUST have the following keys:
    - "component_type": A general classification (e.g., "Resistor", "Inductor", "Flux Stabilizer").
    - "model_number": The specific model number of the component.
    - "manufacturer": The company that produces the component.
    - "properties": A list of objects, where each object represents a technical property.
      Each property object must have a "name" and a "value" (including units).
    - "features": A list of key features as strings.
    
    Do not include any introductory text, explanations, or markdown formatting
    around the final JSON object. Your entire response must be only the JSON object itself.
    """

def extract_knowledge_from_datasheet(filepath):
    """
    The main worker function. It loads a datasheet, sends it to the OpenAI API,
    and parses the structured JSON response.
    """
    print(f"-> Loading raw markdown from: {os.path.basename(filepath)}")
    markdown_content = load_file(filepath)
    if not markdown_content:
        return None

    system_prompt = build_extraction_prompt()
    
    # For starting, gpt-4.1-nano is fast and cost-effective.        
    model_select="gpt-4.1-nano"
    print(f"-> Sending request to OpenAI API (Model: {model_select})...")
    try:
        # This is the core API call to OpenAI's Chat Completions endpoint
        response = client.chat.completions.create(
            # Force the model to return a valid JSON object.
            response_format={"type": "json_object"},
            model = model_select,
            # The 'messages' list defines the conversation with the LLM.
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": markdown_content} # The raw datasheet is the user's message.
            ]
        )
        
        # The response content is a guaranteed JSON string thanks to response_format.
        response_content = response.choices[0].message.content
        extracted_data = json.loads(response_content)
        
        print("-> Successfully extracted and parsed JSON data.")
        return extracted_data

    except Exception as e:
        print(f"An error occurred during the OpenAI API call: {e}")
        return None


if __name__ == "__main__":
    # Process all MD files in data/raw_markdown
    if not os.path.exists(RAW_DATA_DIR):
        print(f"ERROR: Raw data directory not found at '{RAW_DATA_DIR}'")
        print("Please create it and add your markdown datasheets.")
    else:
        datasheet_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.md')]
        
        if not datasheet_files:
            print(f"No markdown files found in '{RAW_DATA_DIR}'.")
        else:
            print(f"Found {len(datasheet_files)} datasheets to process.")
            
            for filename in datasheet_files:
                print(f"\n--- Processing: {filename} ---")
                datasheet_filepath = os.path.join(RAW_DATA_DIR, filename)
                
                # Perform the extraction for the current file
                structured_data = extract_knowledge_from_datasheet(datasheet_filepath)

                if structured_data:
                    # Define the output path for the new JSON file
                    output_filename = os.path.splitext(filename)[0] + "_extracted.json"
                    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
                    
                    # Save the structured data
                    save_json(output_filepath, structured_data)
                    
                    print(f"Success! Extracted knowledge saved to: {output_filepath}")

            print("\nAll files processed.")