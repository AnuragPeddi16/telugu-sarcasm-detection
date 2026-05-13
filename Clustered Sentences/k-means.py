import json
import numpy as np
from sklearn.cluster import KMeans
import os

# Load the data with embeddings
print("Loading data...")
with open('5000-samples-with-embeddings.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

# Extract embeddings
print("Extracting embeddings...")
embeddings = np.array([sample['embeddings'] for sample in data])
print(f"Embeddings shape: {embeddings.shape}")

# Create output directory if it doesn't exist
os.makedirs('Outputs', exist_ok=True)

# Perform K-Means clustering for K values from 3 to 20
for k in range(3, 21):
    print(f"\n{'='*50}")
    print(f"Clustering with K={k}...")
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Create output data with cluster assignments
    clustered_data = []
    for i, sample in enumerate(data):
        output_sample = {
            "utterance": sample['utterance'],
            "context": sample.get('context'),  # Use get() in case context is missing
            "metadata": sample.get('metadata'),  # Use get() in case metadata is missing
            "label": sample['label'],
            "Cluster ID": int(cluster_labels[i])  # Convert numpy.int64 to Python int
        }
        clustered_data.append(output_sample)
    
    # Save to file
    output_filename = f'Outputs/Clustered_5000_K_{k}.json'
    with open(output_filename, 'w') as f:
        json.dump(clustered_data, f, indent=2)
    
    print(f"Saved clustered data to {output_filename}")
    
    # Print cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Cluster distribution for K={k}:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} samples ({count/len(data)*100:.2f}%)")

print(f"\n{'='*50}")
print("All clustering complete!")
print(f"Results saved in the 'Outputs' directory")
