import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("archita091234/fine-tuned-translation")
model = AutoModelForSeq2SeqLM.from_pretrained("archita091234/fine-tuned-translation")
model = model.to(device)
print(f"Model loaded on {device}")

def translate_text(text, max_length=1024):
    """Translate a single text from English to Telugu"""
    if not text or not isinstance(text, str):
        return text
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate translation
    outputs = model.generate(**inputs, max_length=max_length)
    
    # Decode and return
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

def process_dataset(input_file, output_file):
    """Process the entire dataset and create translated version"""
    print(f"Loading dataset from {input_file}...")
    
    # Load input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    translated_data = []
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Processing {len(data)} entries...")
    for idx, entry in enumerate(tqdm(data), start=1):
        translated_entry = {
            "input": entry.get("utterance", ""),
            "utterance": translate_text(entry.get("utterance", "")),
            "label": entry.get("label", ""),
            "metadata": entry.get("metadata", {})
        }
        
        # Handle context if present
        if "context" in entry and entry["context"]:
            translated_entry["input context"] = entry["context"]
            translated_entry["context"] = [
                translate_text(context_text) for context_text in entry["context"]
            ]
        
        translated_data.append(translated_entry)

        if idx % 100 == 0:
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{idx}.json")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            print(f"Checkpoint saved: {checkpoint_file}")
    
    # Save output JSON
    print(f"Saving translated dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    
    print(f"Translation complete! Processed {len(translated_data)} entries.")

if __name__ == "__main__":
    input_file = "5000-samples.json"
    output_file = "/ssd_scratch/phani_ssd/model_out/translations_archita.json"
    
    process_dataset(input_file, output_file)
