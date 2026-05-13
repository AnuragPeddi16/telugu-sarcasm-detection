import torch
import json
import warnings
import os
import threading
import traceback
from queue import Queue
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
model_name = "prajdabre/rotary-indictrans2-en-indic-1B"
input_file = "5000-samples.json"
output_file = "/ssd_scratch/phani_ssd/model_out/translations_rope_indictrans2.json"
output_dir = "/ssd_scratch/phani_ssd/model_out"
checkpoint_dir = os.path.join(output_dir, "checkpoints")
error_log_file = os.path.join(output_dir, "translation_errors.json")
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"CUDA available: {torch.cuda.is_available()} | GPU count: {NUM_GPUS}")
if torch.cuda.is_available():
    for i in range(NUM_GPUS):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

# Initialize models on each GPU
print("\nLoading models on all GPUs...")
models = []
tokenizers = []
processors = []
error_counts = [0] * NUM_GPUS  # Track errors per GPU

for gpu_id in range(NUM_GPUS):
    print(f"Loading model on GPU {gpu_id}...")
    
    try:
        # Create processor and tokenizer for each GPU
        ip = IndicProcessor(inference=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        if torch.cuda.is_available():
            model = model.to(f"cuda:{gpu_id}")
        
        model.eval()
        
        models.append(model)
        tokenizers.append(tokenizer)
        processors.append(ip)
        
        print(f"✓ Model loaded on GPU {gpu_id}")
    except Exception as e:
        print(f"✗ Failed to load model on GPU {gpu_id}: {str(e)}")
        raise

print(f"\n✓ All {NUM_GPUS} models loaded successfully!")

# Global error tracking
translation_errors = []
error_lock = threading.Lock()

def translate_single_text(text, gpu_id, text_idx, src_lang="eng_Latn", tgt_lang="tel_Telu"):
    """Translate a single text on a specific GPU with error handling"""
    if not text or not isinstance(text, str) or text.strip() == "":
        return text
    
    try:
        model = models[gpu_id]
        tokenizer = tokenizers[gpu_id]
        ip = processors[gpu_id]
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
        # Preprocess single text
        batch = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        batch = tokenizer(
            batch, 
            padding="longest", 
            truncation=True, 
            max_length=2048, 
            return_tensors="pt"
        )
        
        # Move batch tensors to specific GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.inference_mode():
            outputs = model.generate(
                **batch,
                num_beams=10,
                length_penalty=1.5,
                repetition_penalty=2.0,
                num_return_sequences=1,
                max_new_tokens=1024,
                early_stopping=True,
                use_cache=False
            )
        
        outputs = tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        
        outputs = ip.postprocess_batch(outputs, lang=tgt_lang)
        
        # Clear cache to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return outputs[0]
        
    except Exception as e:
        error_msg = f"[ERROR on GPU {gpu_id}] {str(e)}"
        error_detail = {
            "text_index": text_idx,
            "gpu_id": gpu_id,
            "original_text": text[:100] + "..." if len(text) > 100 else text,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        with error_lock:
            translation_errors.append(error_detail)
            error_counts[gpu_id] += 1
        
        print(f"\n⚠ Error at index {text_idx} on GPU {gpu_id}: {str(e)}")
        
        # Return error marker so we can identify failed translations
        return f"[TRANSLATION_ERROR: {str(e)[:100]}]"

def process_texts_multi_gpu(all_texts, desc="Processing"):
    """Process texts across multiple GPUs with 1 text per GPU at a time"""
    results = [None] * len(all_texts)
    result_lock = threading.Lock()
    
    # Progress bar
    pbar = tqdm(total=len(all_texts), desc=desc)
    
    def worker(gpu_id):
        """Worker function for each GPU - processes one text at a time"""
        while True:
            try:
                text_idx = task_queue.get_nowait()
            except:
                break
            
            text = all_texts[text_idx]
            translated = translate_single_text(text, gpu_id, text_idx)
            
            with result_lock:
                results[text_idx] = translated
                pbar.update(1)
            
            task_queue.task_done()
    
    # Create queue and add all text indices
    task_queue = Queue()
    for i in range(len(all_texts)):
        task_queue.put(i)
    
    # Start threads for each GPU
    threads = []
    for gpu_id in range(NUM_GPUS):
        t = threading.Thread(target=worker, args=(gpu_id,))
        t.start()
        threads.append(t)
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    pbar.close()
    
    return results

# Load the dataset
print(f"Loading dataset from {input_file}...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Dataset loaded with {len(data)} entries")

# Translate the dataset
print("\nStarting translation...")
print(f"Using {NUM_GPUS} GPUs - Processing 1 text per GPU (safe mode to avoid OOM)")
print(f"Effective parallel processing: {NUM_GPUS} texts at once")
translated_data = []

# Step 1: Translate all utterances
print("\nStep 1: Translating utterances...")
utterances = [entry.get("utterance", "") for entry in data]
translated_utterances = process_texts_multi_gpu(utterances, desc="Translating utterances")

# Step 2: Collect all context texts and translate
print("\nStep 2: Translating contexts...")
all_contexts = []
context_structures = []

for entry in data:
    if "context" in entry and entry["context"]:
        if isinstance(entry["context"], list):
            context_structures.append(("list", len(entry["context"])))
            all_contexts.extend(entry["context"])
        elif isinstance(entry["context"], str):
            context_structures.append(("string", 1))
            all_contexts.append(entry["context"])
        else:
            context_structures.append(("none", 0))
    else:
        context_structures.append(("none", 0))

if all_contexts:
    translated_contexts = process_texts_multi_gpu(all_contexts, desc="Translating contexts")
else:
    translated_contexts = []

# Step 3: Reconstruct the dataset
print("\nStep 3: Reconstructing dataset...")
context_idx = 0
for i, entry in enumerate(tqdm(data, desc="Reconstructing")):
    translated_entry = {}
    
    # Store original input fields
    if "utterance" in entry:
        translated_entry["input"] = entry["utterance"]
    
    # Add translated utterance
    if "utterance" in entry and entry["utterance"]:
        translated_entry["utterance"] = translated_utterances[i]
    
    # Reconstruct context
    struct_type, struct_len = context_structures[i]
    if struct_type == "list":
        translated_entry["input context"] = entry["context"]
        translated_entry["context"] = translated_contexts[context_idx:context_idx + struct_len]
        context_idx += struct_len
    elif struct_type == "string":
        translated_entry["input context"] = entry["context"]
        translated_entry["context"] = translated_contexts[context_idx]
        context_idx += 1
    
    # Copy label and metadata
    if "label" in entry:
        translated_entry["label"] = entry["label"]
    if "metadata" in entry:
        translated_entry["metadata"] = entry["metadata"]
    
    translated_data.append(translated_entry)
    
    # Save checkpoints every 100 entries
    if (i + 1) % 1000 == 0:
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{i + 1:06d}.json")
        print(f"\nSaving checkpoint at entry {i + 1} to {checkpoint_file}...")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

# Save the translated dataset
print(f"\nSaving final translated dataset to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=2)

print("Translation complete!")
print(f"Translated dataset saved to {output_file}")

# Save error log if there were any errors
if translation_errors:
    print(f"\n⚠ {len(translation_errors)} translation errors occurred")
    print(f"Error breakdown by GPU:")
    for gpu_id in range(NUM_GPUS):
        print(f"  GPU {gpu_id}: {error_counts[gpu_id]} errors")
    
    print(f"\nSaving error log to {error_log_file}...")
    with open(error_log_file, 'w', encoding='utf-8') as f:
        json.dump(translation_errors, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Error log saved. You can rerun failed translations using this log.")
    
    # Count entries with errors in final output
    error_entries = sum(1 for entry in translated_data 
                       if ("[TRANSLATION_ERROR" in str(entry.get("utterance", "")) or 
                           "[TRANSLATION_ERROR" in str(entry.get("context", ""))))
    print(f"✓ {error_entries} entries marked with [TRANSLATION_ERROR] in output")
else:
    print(f"\n✓ All translations completed successfully with no errors!")

# Print a sample translation
if translated_data:
    print("\n=== Sample Translation ===")
    sample = translated_data[0]
    if "input" in sample:
        print(f"Original utterance: {sample['input']}")
    if "utterance" in sample:
        print(f"Translated utterance: {sample['utterance']}")
    if 'input context' in sample and sample['input context']:
        print(f"\nOriginal context[0]: {sample['input context'][0] if isinstance(sample['input context'], list) else sample['input context']}")
    if 'context' in sample and sample['context']:
        print(f"Translated context[0]: {sample['context'][0] if isinstance(sample['context'], list) else sample['context']}")

print("\n" + "="*80)
print(f"Summary:")
print(f"  Total entries processed: {len(translated_data)}")
print(f"  Translation errors: {len(translation_errors)}")
print(f"  Success rate: {((len(translated_data) * 2 - len(translation_errors)) / (len(translated_data) * 2) * 100):.2f}%")
print("="*80)
