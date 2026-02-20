import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

def translate_text(text, tokenizer, model):
    """Translate English text to Telugu using instruction-based prompting"""
    messages = [
        {
            "role": "system",
						"content": (
                "You are a professional translator who can translate English text to Telugu."
                "Translate the given English text to natural and fluent Telugu."
                "Preserve the tone, sarcasm."
                "Don't add any extra information or explanation, just provide the translation."
						)
				},
        # Example 1
				{
						"role": "user",
						"content": f"Translate the following English text to Telugu:\nlooks like you have written the exam very good today"
				},
        {
            "role": "assistant",
            "content": "ఈరోజు నువ్వు పరీక్ష చాలా బాగా రాసినట్లున్నావు."
				},
				# Example 2
				{
						"role": "user",
						"content": f"Translate the following English text to Telugu:\nwhat a great climate today"
				},
				{
						"role": "assistant",
						"content": "ఈరోజు ఎంత అద్భుతమైన వాతావరణం ఉంది"
				},
				# Actual input
				{
						"role": "user",
						"content": f"Translate the following English text to Telugu:\n{text}"
				}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0
        )
    
    translation = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()
    
    return translation

def translate_dataset(input_file, output_file):
    """Translate the entire dataset"""
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Translating {len(data)} entries...")
    translated_data = []
    
    for entry in tqdm(data, desc="Translating"):
        translated_entry = {
            "input": entry.get("utterance", ""),
            "label": entry.get("label", "")
        }
        
        # Translate utterance
        if "utterance" in entry and entry["utterance"]:
            translated_entry["utterance"] = translate_text(
                entry["utterance"], 
                tokenizer, 
                model
            )
        
        # Translate context if it exists
        if "context" in entry and entry["context"]:
            translated_entry["input context"] = entry["context"]
            translated_entry["context"] = []
            for context_text in entry["context"]:
                if context_text:
                    translated_context = translate_text(
                        context_text, 
                        tokenizer, 
                        model
                    )
                    translated_entry["context"].append(translated_context)
        
        # Preserve metadata if it exists
        if "metadata" in entry:
            translated_entry["metadata"] = entry["metadata"]
        
        translated_data.append(translated_entry)
    
    # Save translated dataset
    print(f"Saving translated dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    
    print("Translation complete!")

if __name__ == "__main__":
    input_file = "sampled_data 5.json"
    output_file = "translated_data_telugu.json"
    
    translate_dataset(input_file, output_file)