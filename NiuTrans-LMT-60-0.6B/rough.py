import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("NiuTrans/LMT-60-0.6B", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("NiuTrans/LMT-60-0.6B", torch_dtype="auto", device_map="auto")

# Move model to GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

def translate_text(text, source_lang="English", target_lang="Telugu"):
    """Translate text using chat template"""
    # Create prompt using chat template
    prompt = f"Translate the following text from {source_lang} to {target_lang}.\n{source_lang}: {text}\n{target_lang}:"
    
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text_input, return_tensors="pt", max_length=2048, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_beams=5,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the translated part (after the target language marker)
    if f"{target_lang}:" in translation:
        translation = translation.split(f"{target_lang}:")[-1].strip()
    
    return translation

def translate_dataset(input_file, output_file):
    """Translate the entire dataset"""
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    translated_data = []
    total = len(data)
    
    for idx, entry in enumerate(data):
        print(f"Processing entry {idx + 1}/{total}...")
        
        translated_entry = {
            "original_utterance": entry["utterance"],
            "translated_utterance": translate_text(entry["utterance"]),
            "label": entry["label"]
        }
        
        # Translate context if it exists
        if "context" in entry and entry["context"]:
            translated_entry["original_context"] = entry["context"]
            translated_entry["translated_context"] = [
                translate_text(context_text) for context_text in entry["context"]
            ]
        
        # Preserve metadata if it exists
        if "metadata" in entry:
            translated_entry["metadata"] = entry["metadata"]
        
        translated_data.append(translated_entry)
        
        # Save progress every 10 entries
        if (idx + 1) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            print(f"Progress saved: {idx + 1}/{total} entries")
    
    # Final save
    print(f"Saving translated dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    
    print("Translation complete!")

if __name__ == "__main__":
    input_file = "sampled_data.json"
    output_file = "/ssd_scratch/phani_ssd/model_out/translated_output.json"
    
    translate_dataset(input_file, output_file)