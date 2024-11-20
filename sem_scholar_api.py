import requests
import time
from transformers import AutoTokenizer, AutoModel
import torch
import re

# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("globuslabs/ScholarBERT")
model = AutoModel.from_pretrained("globuslabs/ScholarBERT")

def get_bert_embedding(text):
    """
    Generate BERT embedding for the given text.

    Parameters:
    text (str): The input text to generate embedding for.

    Returns:
    torch.Tensor: The BERT embedding vector.
    """
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    
    # Generate BERT embedding
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token embedding as the sentence embedding
    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    
    return embedding.numpy().tolist()

def get_paper(paper_title):
    """
    Retrieve information about a scientific paper from the Semantic Scholar API
    and generate BERT embedding for its abstract.

    Parameters:
    paper_title (str): The title of the paper to search for.

    Returns:
    dict or None: A dictionary containing information about the paper if found,
                  including fields such as title, abstract, open access status,
                  paper ID, and BERT embedding. Returns None if the request fails or no paper is found.
    """
    paper_title= re.sub(r"[^\t\r\n\x20-\x7E]+", ' ', paper_title) 
    print(paper_title)
    # Define the API endpoint URL
    url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"

    # Define the query parameters
    query_params = {"fields": "title,abstract,isOpenAccess,openAccessPdf,paperId", "query": paper_title}

    # Send the API request
    response = requests.get(url, params=query_params)

    # Check response status
    if response.status_code == 200:
        response_data = response.json()['data'][0]
        print('Generating BERT embedding for paper abstract')
        if response_data['abstract']:
            embedding = get_bert_embedding(response_data['abstract'])
            response_data['embedding'] = embedding
        else:
            print(f"No abstract found for paper: {paper_title}, using title embedding instead.")
            response_data['embedding'] = get_bert_embedding(paper_title)
        return response_data
    if response.status_code == 429:
        print("Rate limit exceeded. Wait for one minute before making another request.")
        time.sleep(5)
        return get_paper(paper_title)
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return None

# Example usage
if __name__ == "__main__":
    paper_title = "Spatial-Temporal Graph Convolutional Network for Video-Based Person Re-Identification"
    result = get_paper(paper_title)
    if result:
        print(result)