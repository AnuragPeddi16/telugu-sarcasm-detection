from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
import os
from itertools import combinations
import random
from tqdm import tqdm

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_similarity(tokenizer, model, sent1, sent2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Batch encode
    encoded_input = tokenizer([sent1, sent2], padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return F.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()

def calculate_intra_cluster_similarity(clusters, tokenizer, model, device, sample_size=50):
    """Calculate average similarity within clusters"""
    similarities = []
    
    for cluster_id, utterances in clusters.items():
        if len(utterances) < 2:
            continue
        
        # Sample pairs if cluster is large
        if len(utterances) > sample_size:
            sampled_utterances = random.sample(utterances, sample_size)
        else:
            sampled_utterances = utterances
        
        # Calculate similarity for all pairs within cluster
        pairs = list(combinations(sampled_utterances, 2))
        if len(pairs) > 100:  # Limit pairs to avoid too many computations
            pairs = random.sample(pairs, 100)
        
        for sent1, sent2 in pairs:
            sim = get_similarity(tokenizer, model, sent1, sent2, device)
            similarities.append(sim)
    
    return sum(similarities) / len(similarities) if similarities else 0.0

def calculate_inter_cluster_similarity(clusters, tokenizer, model, device, sample_size=30):
    """Calculate average similarity between different clusters"""
    similarities = []
    cluster_ids = list(clusters.keys())
    
    if len(cluster_ids) < 2:
        return 0.0
    
    # Sample cluster pairs
    cluster_pairs = list(combinations(cluster_ids, 2))
    if len(cluster_pairs) > 20:  # Limit cluster pairs
        cluster_pairs = random.sample(cluster_pairs, 20)
    
    for cluster1_id, cluster2_id in cluster_pairs:
        utterances1 = clusters[cluster1_id]
        utterances2 = clusters[cluster2_id]
        
        # Sample utterances from each cluster
        sample1 = random.sample(utterances1, min(sample_size, len(utterances1)))
        sample2 = random.sample(utterances2, min(sample_size, len(utterances2)))
        
        # Calculate similarity between samples
        for sent1 in sample1[:10]:  # Limit to avoid too many comparisons
            for sent2 in sample2[:10]:
                sim = get_similarity(tokenizer, model, sent1, sent2, device)
                similarities.append(sim)
    
    return sum(similarities) / len(similarities) if similarities else 0.0

def benchmark_clusters(file_path, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Benchmark a single cluster file"""
    print(f"\nProcessing: {file_path}")
    
    # Extract K value from filename
    k_value = int(file_path.split('_K_')[1].split('.')[0])
    
    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Organize by cluster
    clusters = {}
    for item in data:
        cluster_id = item['Cluster ID']
        utterance = item['utterance']
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(utterance)
    
    print(f"K={k_value}, Total clusters: {len(clusters)}, Total samples: {len(data)}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Calculate metrics
    print("Calculating intra-cluster similarity...")
    intra_sim = calculate_intra_cluster_similarity(clusters, tokenizer, model, device)
    
    print("Calculating inter-cluster similarity...")
    inter_sim = calculate_inter_cluster_similarity(clusters, tokenizer, model, device)
    
    # Review: Good clustering has high intra-cluster and low inter-cluster similarity
    # Thresholds: intra > 0.5, inter < 0.4
    review = "Pass" if intra_sim > 0.5 and inter_sim < 0.4 else "Fail"
    
    return {
        "K": k_value,
        "intra-cluster similarity": round(intra_sim, 4),
        "inter-cluster similarity": round(inter_sim, 4),
        "Review": review
    }

def main():
    # Models to test
    models = [
        'sentence-transformers/all-MiniLM-L6-v2',
        'sentence-transformers/all-mpnet-base-v2',
        'sentence-transformers/paraphrase-MiniLM-L6-v2'
    ]
    
    # Get all cluster files
    outputs_dir = 'Outputs'
    cluster_files = sorted([f for f in os.listdir(outputs_dir) if f.startswith('Clustered_5000_K_')])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Total cluster files to process: {len(cluster_files)}")
    
    # Run benchmarking for each model
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Benchmarking with model: {model_name}")
        print(f"{'='*80}")
        
        benchmarks = []
        
        for cluster_file in tqdm(cluster_files, desc=f"Processing with {model_name.split('/')[-1]}"):
            file_path = os.path.join(outputs_dir, cluster_file)
            result = benchmark_clusters(file_path, model_name, device)
            benchmarks.append(result)
            print(f"K={result['K']}: Intra={result['intra-cluster similarity']:.4f}, Inter={result['inter-cluster similarity']:.4f}, {result['Review']}")
        
        # Sort by K value
        benchmarks = sorted(benchmarks, key=lambda x: x['K'])
        
        # Extract model name for filename
        model_filename = model_name.split('/')[-1]  # e.g., 'all-MiniLM-L6-v2'
        output_filename = f'benchmark_{model_filename}.json'
        
        # Save results
        with open(output_filename, 'w') as f:
            json.dump(benchmarks, f, indent=2)
        
        print(f"\n✓ Benchmarks saved to {output_filename}")
        print(f"Total files processed: {len(benchmarks)}")
    
    print(f"\n{'='*80}")
    print("All benchmarking complete!")

if __name__ == "__main__":
    main()
