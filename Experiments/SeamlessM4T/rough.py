import json
import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
from tqdm import tqdm

# Load the model and processor
print("Loading model and processor...")
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")

# Move model to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

def translate_text(text, src_lang="eng", tgt_lang="tel"):
    """Translate text from English to Telugu"""
    if not text or not isinstance(text, str):
        return text
    
    # Process the input text
    inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate translation
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            tgt_lang=tgt_lang,
            num_beams=5,
            # max_length=512,
            max_new_tokens=512,
            early_stopping=True
        )
    
    # Decode the output
    translated_text = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
    return translated_text

def translate_dataset(input_file, output_file):
    """Translate the entire dataset"""
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Translating {len(data)} entries...")
    translated_data = []
    
    for entry in tqdm(data, desc="Translating"):
        translated_entry = {}
        
        # Store original utterance as 'input' and translate to 'utterance'
        if 'utterance' in entry:
            translated_entry['input'] = entry['utterance']
            if entry['utterance']:
                translated_entry['utterance'] = translate_text(entry['utterance'])
            else:
                translated_entry['utterance'] = entry['utterance']
        
        # Store original context as 'input context' and translate to 'context'
        if 'context' in entry and isinstance(entry['context'], list):
            translated_entry['input context'] = entry['context']
            translated_context = []
            for context_item in entry['context']:
                if context_item:
                    translated_context.append(translate_text(context_item))
                else:
                    translated_context.append(context_item)
            translated_entry['context'] = translated_context
        
        # Keep label and add metadata
        if 'label' in entry:
            translated_entry['label'] = entry['label']
        translated_entry['metadata'] = {}
        
        translated_data.append(translated_entry)
        
        # Save periodically (every 100 entries)
        if len(translated_data) % 100 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            print(f"\nSaved progress: {len(translated_data)} entries")
    
    # Final save
    print(f"\nSaving final output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    
    print("Translation complete!")

if __name__ == "__main__":
    input_file = "sampled_data.json"
    output_file = "translations_seamlessm4tv2.json"
    
    translate_dataset(input_file, output_file)
