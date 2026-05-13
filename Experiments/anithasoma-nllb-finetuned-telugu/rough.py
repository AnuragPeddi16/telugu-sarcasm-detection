import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("anithasoma/nllb-finetuned-telugu")
model = AutoModelForSeq2SeqLM.from_pretrained("anithasoma/nllb-finetuned-telugu")

# Move model to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

def translate_text(text, max_length=1024):
    """Translate English text to Telugu"""
    if not text or not isinstance(text, str):
        return text
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
    
    # Decode output
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

def translate_entry(entry):
    """Translate an entry from the dataset following sample_output.json format"""
    translated_entry = {}
    
    # Keep the original English utterance as "input"
    if "utterance" in entry and entry["utterance"]:
        translated_entry["input"] = entry["utterance"]
        # Translate utterance to Telugu
        translated_entry["utterance"] = translate_text(entry["utterance"])
    
    # Translate context if present
    if "context" in entry and isinstance(entry["context"], list):
        # Keep original English context as "input context"
        translated_entry["input context"] = entry["context"]
        
        # Translate context to Telugu
        translated_context = []
        for context_text in entry["context"]:
            if context_text:
                translated_context.append(translate_text(context_text))
            else:
                translated_context.append(context_text)
        translated_entry["context"] = translated_context
    
    # Keep label and metadata
    if "label" in entry:
        translated_entry["label"] = entry["label"]
    
    if "metadata" in entry:
        translated_entry["metadata"] = entry["metadata"]
    else:
        translated_entry["metadata"] = {}
    
    return translated_entry

# Load input data
print("Loading dataset...")
with open("5000-samples.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total entries to translate: {len(data)}")

# Translate all entries
translated_data = []
print("Starting translation...")

for i, entry in enumerate(tqdm(data, desc="Translating")):
    try:
        translated_entry = translate_entry(entry)
        translated_data.append(translated_entry)
        
        # Save progress every 100 entries
        if (i + 1) % 100 == 0:
            with open("/ssd_scratch/phani_ssd/model_out/translations_anithasoma_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
    
    except Exception as e:
        print(f"\nError translating entry {i}: {str(e)}")
        # Keep original entry if translation fails
        translated_data.append(entry)

# Save final output
print("Saving translated dataset...")
with open("/ssd_scratch/phani_ssd/model_out/translations_anithasoma.json", "w", encoding="utf-8") as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=2)

print("Translation complete!")
print(f"Output saved to: /ssd_scratch/phani_ssd/model_out/translations_anithasoma.json")