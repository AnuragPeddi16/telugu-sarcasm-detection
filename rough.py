import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import gc

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B")

# Move model to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

def translate_text(text, src_lang="eng_Latn", tgt_lang="tel_Telu", max_length=1024):
    """Translate text from source language to target language"""
    if not text or text.strip() == "":
        return text
    
    try:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Set the source language
        tokenizer.src_lang = src_lang
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode the generated tokens
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return translated_text
    
    except Exception as e:
        print(f"Error translating text: {str(e)}")
        return text  # Return original text if translation fails

def translate_dataset():
    """Translate the entire dataset"""
    print("Loading dataset...")
    with open('5000-samples.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} entries to translate")
    
    translated_data = []
    
    for i, entry in enumerate(tqdm(data, desc="Translating entries")):
        translated_entry = entry.copy()
        
        # Translate utterance
        if 'utterance' in entry and entry['utterance']:
            translated_entry['utterance'] = translate_text(entry['utterance'])
        
        # Translate context if it exists
        if 'context' in entry and entry['context']:
            translated_context = []
            for context_text in entry['context']:
                if context_text:  # Only translate non-empty context
                    translated_context.append(translate_text(context_text))
                else:
                    translated_context.append(context_text)
            translated_entry['context'] = translated_context
        
        translated_data.append(translated_entry)
    
    # Create output in the exact format from sample_output.json with ALL entries
    print("Creating output in sample_output.json format...")
    formatted_output = []
    for i, (original, translated) in enumerate(zip(data, translated_data)):
        formatted_entry = {
            "input": original.get('utterance', ''),
            "utterance": translated.get('utterance', ''),
            "input context": original.get('context', []),
            "context": translated.get('context', []),
            "label": original.get('label', ''),
            "metadata": original.get('metadata', {})
        }
        formatted_output.append(formatted_entry)
    
    # Save all entries in the sample_output.json format
    output_file = '/ssd_scratch/phani_ssd/model_out/translations_nllb_200.json'
    print(f"Saving all {len(formatted_output)} translated entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_output, f, ensure_ascii=False, indent=2)
    
    print(f"Translation complete! All entries saved to {output_file}")

if __name__ == "__main__":
    translate_dataset()