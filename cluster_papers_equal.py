import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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

def equally_divide_papers(embeddings, titles, n_groups):
    """Divide papers into equal-sized groups based on their embeddings."""
    # Normalize the embeddings
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    
    # Perform initial K-means clustering
    kmeans = KMeans(n_clusters=n_groups, random_state=42)
    kmeans.fit(normalized_embeddings)
    
    # Calculate cosine similarity between papers and cluster centers
    similarities = cosine_similarity(normalized_embeddings, kmeans.cluster_centers_)
    
    # Sort papers based on their similarity to cluster centers
    sorted_indices = np.argsort(similarities.max(axis=1))
    
    # Divide papers into equal-sized groups
    group_size = len(embeddings) // n_groups
    remainder = len(embeddings) % n_groups
    
    groups = []
    start = 0
    for i in range(n_groups):
        end = start + group_size + (1 if i < remainder else 0)
        groups.append(sorted_indices[start:end])
        start = end
    
    # Assign group labels
    labels = np.zeros(len(embeddings), dtype=int)
    for i, group in enumerate(groups):
        labels[group] = i
    
    return labels

def save_clusters_to_csv(titles, cluster_labels, output_file):
    """Save paper titles and their cluster labels to a CSV file, sorted by group number."""
    df = pd.DataFrame({
        'Paper Title': titles,
        'Group': cluster_labels
    })
    
    # Sort the DataFrame by the 'Group' column
    df_sorted = df.sort_values(by='Group')
    
    # Save the sorted DataFrame to CSV
    df_sorted.to_csv(output_file, index=False)
    print(f"Grouping results saved to {output_file}")

def main():
    # Load paper data
    papers = load_paper_data('output_file.json')
    
    # Extract embeddings and titles
    embeddings, titles = extract_embeddings_and_titles(papers)
    
    if len(embeddings) == 0:
        print("No valid embeddings found in the data.")
        return
    
    # Set the number of groups
    n_groups = 4  # You can adjust this number as needed
    
    # Perform equal division of papers
    group_labels = equally_divide_papers(embeddings, titles, n_groups)
    
    # Save results to CSV
    save_clusters_to_csv(titles, group_labels, 'paper_groups_equal_sorted.csv')

if __name__ == "__main__":
    main()