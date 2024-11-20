import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def load_paper_data(file_path):
    """Load paper data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_embeddings_and_titles(papers):
    """Extract embeddings and titles from paper data."""
    embeddings = []
    titles = []
    for paper in papers:
        if paper['Embedding'] != 'N/A':
            embeddings.append(paper['Embedding'])
            titles.append(paper['Retrieved Title'])
    return np.array(embeddings), titles

def load_clusters(file_path):
    """Load cluster assignments from a CSV file."""
    df = pd.read_csv(file_path)
    return df['Paper Title'].tolist(), df['Group'].tolist()

def visualize_clusters(embeddings, titles, groups):
    """Visualize clusters using t-SNE."""
    # Normalize the embeddings
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(normalized_embeddings)

    # Create a scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=groups, cmap='viridis')
    plt.colorbar(scatter)

    # Add labels for some points (adjust n for more or fewer labels)
    n = min(20, len(titles))
    for i in np.random.choice(len(titles), n, replace=False):
        plt.annotate(titles[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)

    plt.title('Paper Clusters Visualization')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.tight_layout()
    plt.savefig('cluster_visualization.png', dpi=300)
    plt.close()
    print("Cluster visualization saved as 'cluster_visualization.png'")

def main():
    # Load paper data
    papers = load_paper_data('output_file.json')
    
    # Extract embeddings and titles
    embeddings, titles = extract_embeddings_and_titles(papers)
    
    if len(embeddings) == 0:
        print("No valid embeddings found in the data.")
        return
    
    # Load cluster assignments
    cluster_titles, groups = load_clusters('paper_groups_equal_sorted.csv')
    
    # Ensure the order of titles matches
    title_to_group = dict(zip(cluster_titles, groups))
    groups = [title_to_group[title] for title in titles]
    
    # Visualize clusters
    visualize_clusters(embeddings, titles, groups)

if __name__ == "__main__":
    main()