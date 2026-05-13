import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("NiuTrans/LMT-60-1.7B")
model = AutoModelForCausalLM.from_pretrained("NiuTrans/LMT-60-1.7B")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Using device: {device}")

def translate_text(text, label, source_lang="English", target_lang="Telugu"):
    """
    Translate text using instruction-based prompting with label-specific instructions
    """
    if not text or text.strip() == "":
        return text
    
    # Create label-specific prompt
    if label == "SARCASTIC":
        prompt = f"""You are a professional translator from {source_lang} to {target_lang}.

Below are reference examples showing how sarcastic English sentences must be translated into {target_lang} while preserving sarcasm.

Examples:

Example 1:
Input ({source_lang}): looks like you have written the exam very good today.
Output ({target_lang}): ఈరోజు నువ్వు పరీక్ష చాలా బాగా రాసినట్లున్నావు.

Example 2:
Input ({source_lang}): what a great climate today
Output ({target_lang}): ఈరోజు ఎంత అద్భుతమైన వాతావరణం ఉంది.

Example 3:
Input ({source_lang}): Oh great, another meeting!
Output ({target_lang}): ఓహ్ అద్భుతం, మరో సమావేశం!

Now translate the following text from {source_lang} to {target_lang}:

{source_lang}: {text}
{target_lang}:"""

    else:  # NOT_SARCASTIC
        prompt = f"""You are a professional translator from {source_lang} to {target_lang}.

Below are reference examples showing how normal, non-sarcastic English sentences must be translated into {target_lang}.

Examples:

Example 1:
Input ({source_lang}): Today there was an exam. In that exam, I wasn't able to write anything. And my friend noticed that. After finishing the exam, when we both came out of the examination hall
Output ({target_lang}): ఈరోజు పరీక్ష ఉంది. ఆ పరీక్షలో, నేను ఏమీ రాయలేకపోయాను. నా స్నేహితుడు దానిని గమనించాడు. పరీక్ష ముగిసిన తర్వాత, మేము పరీక్ష హాల్ నుండి బయటకు వచ్చాము.

Example 2:
Input ({source_lang}): I wanted to finish an important work today for which I have to go out, I got ready and opened the door to leave the house, then suddenly rain started.
Output ({target_lang}): నేను ఈ రోజు బయటికి వెళ్ళి ఒక ముఖ్యమైన పనిని పూర్తిచేయాలనుకున్నాను, నేను సిద్ధంగా ఉండి ఇంటి నుండి బయలుదేరడానికి తలుపు తెరిచాను, అప్పుడు అకస్మాత్తుగా వర్షం ప్రారంభమైంది.

Now translate the following text from {source_lang} to {target_lang}:

{source_lang}: {text}
{target_lang}:"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    inputs = inputs.to(device)
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_beams=5,
            temperature=0.7,
            do_sample=True,
            top_p=0.8,
            top_k=20,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the translation part (after the target language marker)
    if f"{target_lang}:" in translated_text:
        translated_text = translated_text.split(f"{target_lang}:")[-1].strip()
    
    return translated_text

def translate_dataset(input_file, output_file):
    """
    Translate the entire dataset from sampled_data.json with label-aware translation
    """
    # Load the dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Translate each entry
    translated_data = []
    print(f"Translating {len(data)} entries...")
    
    for entry in tqdm(data):
        translated_entry = entry.copy()
        
        # Get label for context-aware translation
        label = entry.get('label', 'NOT_SARCASTIC')
        
        # Translate utterance
        if 'utterance' in entry and entry['utterance']:
            translated_entry['utterance_telugu'] = translate_text(
                entry['utterance'], 
                label
            )
        
        # Translate context list if present
        if 'context' in entry and entry['context']:
            translated_context = []
            for context_text in entry['context']:
                translated_context.append(
                    translate_text(context_text, label)
                )
            translated_entry['context_telugu'] = translated_context
        
        translated_data.append(translated_entry)
    
    # Save translated dataset
    print(f"Saving translated dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    
    print("Translation complete!")

if __name__ == "__main__":
    input_file = "sampled_data.json"
    output_file = "/ssd_scratch/phani_ssd/model_out/translated_output.json"
    
    translate_dataset(input_file, output_file)