import json
import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm
import torch

# Load model and tokenizer
print("Loading model and tokenizer...")
model_checkpoint = "gvaishnavi/bilingual-translation"
tokenizer = MBart50TokenizerFast.from_pretrained(model_checkpoint)
model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Using device: {device}")

def translate_batch(texts, batch_size=8):
    """Translate a batch of texts"""
    translations = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i:i + batch_size]
        # Filter empty texts
        valid_texts = [t for t in batch if t and t.strip()]
        
        if not valid_texts:
            translations.extend([''] * len(batch))
            continue
        
        # Tokenize
        inputs = tokenizer(
            valid_texts, 
            return_tensors="pt", 
            padding=True,
            max_length=2048, 
            truncation=True
        ).to(device)
        
        # Generate translations
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=2048,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode
        batch_translations = [
            tokenizer.decode(tokens, skip_special_tokens=True)
            for tokens in generated_tokens
        ]
        
        # Map back to original batch (including empty strings)
        result = []
        valid_idx = 0
        for text in batch:
            if text and text.strip():
                result.append(batch_translations[valid_idx])
                valid_idx += 1
            else:
                result.append(text)
        
        translations.extend(result)
    
    return translations

# Load the JSON data
print("Loading dataset...")
with open('5000-samples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Collect all texts for translation
print("Collecting texts for translation...")
utterances = [item.get('utterance', '') for item in data]

all_contexts = []
context_indices = []
for idx, item in enumerate(data):
    if 'context' in item and isinstance(item['context'], list):
        all_contexts.extend(item['context'])
        context_indices.extend([idx] * len(item['context']))

# Translate utterances
print(f"Translating {len(utterances)} utterances...")
translated_utterances = translate_batch(utterances, batch_size=4)

# Translate contexts
print(f"Translating {len(all_contexts)} context texts...")
translated_contexts = translate_batch(all_contexts, batch_size=4)

# Reconstruct the dataset with input fields
print("Reconstructing dataset...")
translated_data = []
output_dir = "/ssd_scratch/phani_ssd/model_out"
checkpoints_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoints_dir, exist_ok=True)

context_counter = 0
for idx, item in tqdm(enumerate(data), total=len(data), desc="Reconstructing"):
    translated_item = {}
    
    # Add original input
    if 'utterance' in item:
        translated_item['input'] = item['utterance']
    
    # Add translated utterance
    translated_item['utterance'] = translated_utterances[idx]
    
    # Add original input context and translated context
    if 'context' in item and isinstance(item['context'], list):
        context_count = len(item['context'])
        translated_item['input context'] = item['context']
        translated_item['context'] = translated_contexts[context_counter:context_counter + context_count]
        context_counter += context_count
    
    # Add label if present
    if 'label' in item:
        translated_item['label'] = item['label']
    
    # Add metadata if present
    if 'metadata' in item:
        translated_item['metadata'] = item['metadata']
    else:
        translated_item['metadata'] = {}
    
    translated_data.append(translated_item)

    # Save incremental checkpoint after every 100 samples
    if (idx + 1) % 100 == 0:
        checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_{idx + 1}.json")
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        print(f"Checkpoint saved: {checkpoint_path}")

# Save final checkpoint if the dataset size is not a multiple of 100
if len(translated_data) % 100 != 0:
    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_{len(translated_data)}.json")
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    print(f"Final checkpoint saved: {checkpoint_path}")

# Save the translated data
print("Saving translated dataset...")
with open(os.path.join(output_dir, 'translated_output.json'), 'w', encoding='utf-8') as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=2)

print("Translation complete! Output saved to '/ssd_scratch/phani_ssd/model_out/translated_output.json'")
print(f"Total entries translated: {len(translated_data)}")
