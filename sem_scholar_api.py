import requests
import time


def get_paper_embedding(paper_id):
    """
    Retrieve the embedding vector for a scientific paper from the Semantic Scholar API.

    This function sends a request to the Semantic Scholar API to retrieve the embedding
    vector for a given paper ID.

    Parameters:
    paper_id (str): The ID of the paper for which to retrieve the embedding.

    Returns:
    list or None: A list containing the embedding vector if found, or None if the request fails.
    """
    # Define the API endpoint URL
    url = f"http://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=embedding"

    # Send the API request
    response = requests.get(url)

    # Check response status
    if response.status_code == 200:
        response_data = response.json()
        if 'embedding' in response_data:
            return response_data['embedding']['vector']
        else:
            print(f"No embedding found for paper ID: {paper_id}")
            return None
    elif response.status_code == 429:
        print("Rate limit exceeded. Wait for one minute before making another request.")
        time.sleep(60)
        return get_paper_embedding(paper_id)
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return None

def get_paper(paper_title):
    """
    Retrieve information about a scientific paper from the Semantic Scholar API.

    This function sends a request to the Semantic Scholar API to search for a paper
    based on its title and returns detailed information about the paper if found.

    Parameters:
    paper_title (str): The title of the paper to search for.

    Returns:
    dict or None: A dictionary containing information about the paper if found,
                  including fields such as title, abstract, open access status,
                  and paper ID. Returns None if the request fails or no paper is found.
    """
    # Define the API endpoint URL
    url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"

    # Define the query parameters
    query_params = {"fields": "title,abstract,isOpenAccess,openAccessPdf,paperId", "query": paper_title}

    # Send the API request
    response = requests.get(url, params=query_params)

    # Check response status
    if response.status_code == 200:
        response_data = response.json()['data'][0]
        print('get paper embedding')
        embedding = get_paper_embedding(response_data['paperId'])
        if embedding:
            response_data['embedding'] = embedding
        return response_data
    if response.status_code == 429:
        print("Rate limit exceeded. Wait for one minute before making another request.")
        time.sleep(60)
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