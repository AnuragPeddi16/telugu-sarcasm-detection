import torch
import torch.nn.functional as F
import numpy as np
import json
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_embedding(text, tokenizer, model, first_device):
    """Generate embedding for a single text"""
    batch_dict = tokenizer(text, max_length=8192, truncation=True, return_tensors="pt")
    batch_dict = {k: v.to(first_device) for k, v in batch_dict.items()}
    
    with torch.no_grad():
        outputs = model(**batch_dict)
        embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
    
    return embedding.detach().cpu().numpy()[0].tolist()


# Set up multi-GPU environment
print(f"Available GPUs: {torch.cuda.device_count()}")
assert torch.cuda.is_available(), "CUDA not available"
assert torch.cuda.device_count() >= 4, f"Need 4 GPUs, found {torch.cuda.device_count()}"

# Load model and tokenizer with automatic device mapping across 4 GPUs
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left')
model = AutoModel.from_pretrained(
    'Qwen/Qwen3-Embedding-8B',
    device_map='auto',
    max_memory={0: '9GB', 1: '9GB', 2: '9GB', 3: '9GB'},
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.eval()

first_device = next(iter(model.hf_device_map.values()))
print(f"Model loaded. Device map: {model.hf_device_map}")

# Load JSON data
print("\nLoading JSON data...")
with open('5000-samples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

# Process each sample and add embeddings
print("\nGenerating embeddings...")
for i, sample in enumerate(tqdm(data)):
    utterance = sample['utterance']
    embedding = get_embedding(utterance, tokenizer, model, first_device)
    sample['embeddings'] = embedding
    
    # Print progress every 500 samples
    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(data)} samples")

# Save updated JSON
output_file = '5000-samples-with-embeddings.json'
print(f"\nSaving embeddings to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Done! Embeddings saved to {output_file}")
print(f"Embedding dimension: {len(data[0]['embeddings'])}")
