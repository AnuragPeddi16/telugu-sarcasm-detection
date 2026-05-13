import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("tencent/HY-MT1.5-1.8B")
model = AutoModelForCausalLM.from_pretrained(
    "tencent/HY-MT1.5-1.8B",
    torch_dtype=torch.float16,
    device_map="auto"
)

def translate_text(text, source_lang="English", target_lang="Telugu", label="NOT_SARCASTIC"):
    """Translate text using default prompt template"""
    if not text or text.strip() == "":
        return text
    
    # Use default prompt for all translations
    prompt = f"""Translate the following segment into {target_lang}, without additional explanation.

{text}"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    
    # Remove token_type_ids if present (not used by this model)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 256,
            num_beams=5,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
    
    # Decode output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Debug: Print first few translations to see what's happening
    print(f"\nOriginal: {text[:100]}")
    print(f"Full output: {full_output[:200]}")
    
    # Extract translation - find text after the prompt
    translation = full_output
    
    # Try to extract just the translation by removing the prompt
    if prompt in translation:
        translation = translation.replace(prompt, "").strip()
    elif text in translation:
        # Split by the original text and take everything after it
        parts = translation.split(text, 1)
        if len(parts) > 1:
            translation = parts[1].strip()
    
    # Clean up any remaining prompt artifacts
    lines = translation.split('\n')
    # Take the first non-empty line that doesn't contain prompt keywords
    for line in lines:
        line = line.strip()
        if line and not any(keyword in line.lower() for keyword in ['translate', 'segment', 'explanation', 'into']):
            translation = line
            break
    
    # If translation is still empty or contains the original text, return the full output
    if not translation or translation == text:
        translation = full_output
    
    print(f"Extracted translation: {translation[:100]}\n")
    
    return translation

def translate_dataset(input_file, output_file):
    """Translate the entire dataset"""
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Translating {len(data)} entries...")
    translated_data = []
    
    for entry in tqdm(data, desc="Translating"):
        translated_entry = entry.copy()
        
        # Get the label for this entry (default to NOT_SARCASTIC if not present)
        label = entry.get('label', 'NOT_SARCASTIC')
        
        # Translate utterance (label parameter kept for compatibility but not used)
        if 'utterance' in entry and entry['utterance']:
            translated_entry['utterance_telugu'] = translate_text(entry['utterance'], label=label)
        
        # Translate context if present
        if 'context' in entry and isinstance(entry['context'], list):
            translated_context = []
            for context_text in entry['context']:
                if context_text:
                    translated_context.append(translate_text(context_text))
                else:
                    translated_context.append(context_text)
            translated_entry['context_telugu'] = translated_context
        
        translated_data.append(translated_entry)
        
        # Save checkpoint every 100 entries
        if len(translated_data) % 100 == 0:
            print(f"\nSaving checkpoint at {len(translated_data)} entries...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
    
    # Save final output
    print(f"\nSaving final output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    
    print("Translation complete!")
    print(f"Total entries processed: {len(translated_data)}")

if __name__ == "__main__":
    input_file = "sampled_data 5.json"
    output_file = "translated_output 5.json"
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, using CPU")
    
    translate_dataset(input_file, output_file)