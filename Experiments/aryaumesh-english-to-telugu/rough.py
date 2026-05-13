import json
import os
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm

def translate_text(text, model, tokenizer, device):
    """Translate a single text from English to Telugu"""
    if not text or not text.strip():
        return text
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    # Decode output
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated

def translate_dataset(input_file, output_file, model_checkpoint="aryaumesh/english-to-telugu"):
    """Translate the entire dataset"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = MBart50TokenizerFast.from_pretrained(model_checkpoint)
    model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Load dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")

    checkpoint_dir = "/ssd_scratch/phani_ssd/model_out/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Translate each entry
    translated_data = []
    for idx, entry in enumerate(tqdm(data, desc="Translating"), start=1):
        translated_entry = {
            "input": entry.get("utterance", ""),
            "utterance": "",
            "input context": entry.get("context", []),
            "context": [],
            "label": entry.get("label", ""),
            "metadata": entry.get("metadata", {})
        }
        
        # Translate utterance
        if entry.get("utterance"):
            translated_entry["utterance"] = translate_text(
                entry["utterance"], 
                model, 
                tokenizer, 
                device
            )
        
        # Translate context list
        if entry.get("context"):
            for context_text in entry["context"]:
                translated_context = translate_text(
                    context_text, 
                    model, 
                    tokenizer, 
                    device
                )
                translated_entry["context"].append(translated_context)
        
        translated_data.append(translated_entry)

        if idx % 100 == 0:
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{idx:05d}.json")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            print(f"Saved checkpoint at sample {idx}: {checkpoint_file}")
    
    # Save translated dataset
    print(f"Saving translated dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    
    print("Translation complete!")
    print(f"Translated {len(translated_data)} entries")

if __name__ == "__main__":
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Translate the dataset
    translate_dataset(
        input_file="5000-samples.json",
        output_file="/ssd_scratch/phani_ssd/model_out/translations_aryaumesh.json"
    )
