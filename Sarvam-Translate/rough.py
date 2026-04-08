import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Load model and tokenizer
model_name = "sarvamai/sarvam-translate"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded successfully!")

def translate_text(text, target_lang="Telugu"):
    """Translate a single text to target language"""
    messages = [
        {"role": "system", "content": f"Translate whole sentence into {target_lang} preserve its meaning not just translating each word literally."},
        {"role": "user", "content": text}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and generate
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.01,
        num_beams=5,
        num_return_sequences=1
    )
    
    # Decode output
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return output_text

# Load input data
print("Loading sampled_data.json...")
with open('sampled_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare output list
output_data = []

# Process each item
print(f"Processing {len(data)} items...")
for item in tqdm(data):
    output_item = {
        "input": item.get("utterance", ""),
        "utterance": "",
        "label": item.get("label", ""),
        "metadata": item.get("metadata", {})
    }
    
    # Translate utterance
    if item.get("utterance"):
        output_item["utterance"] = translate_text(item["utterance"])
    
    # Translate context if it exists
    if "context" in item and item["context"]:
        output_item["input context"] = item["context"]
        output_item["context"] = []
        for context_text in item["context"]:
            translated_context = translate_text(context_text)
            output_item["context"].append(translated_context)
    
    output_data.append(output_item)

# Save output
print("Saving results to translations_sarvam_v1.json...")
with open('Prompt_Outputs/Outputs/translations_sarvam_v1.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("Translation complete!")