import pandas as pd
from sem_scholar_api import get_paper
import time
import pickle

def process_papers(input_file, output_file):
    """
    Process paper information from an input Excel file and save the results to a pickle file.

    This function reads paper titles from an input Excel file, retrieves additional
    information for each paper using the get_paper function, and saves the results
    to a pickle file with added columns for retrieved title, paper ID, abstract,
    and PDF link.

    Parameters:
    input_file (str): The path to the input Excel file containing paper titles.
    output_file (str): The path where the output pickle file will be saved.

    Returns:
    None

    Note:
    The function prints progress messages and the final save location to the console.
    It also includes a 1-second delay between processing each paper to avoid hitting rate limits.
    """
    # Read the input Excel file
    df = pd.read_excel(input_file)

    # Create lists to store results
    titles = []
    paper_ids = []
    abstracts = []
    pdf_links = []
    embeddings = []
    # Process each paper title
    for idx, title in enumerate(df['Paper Title']):
        print(f"#{idx} Processing: {title}")
        paper_info = get_paper(title)

        if paper_info:
            titles.append(paper_info.get('title', 'N/A'))
            paper_ids.append(paper_info.get('paperId', 'N/A'))
            abstracts.append(paper_info.get('abstract', 'N/A'))
            embeddings = paper_info.get('embedding', 'N/A')
            pdf_links.append(paper_info.get('openAccessPdf', {}).get('url', 'N/A'))
        else:
            titles.append('N/A')
            paper_ids.append('N/A')
            abstracts.append('N/A')
            pdf_links.append('N/A')
            embeddings.append('N/A')

        # Add a small delay to avoid hitting rate limits
        time.sleep(1)

    # Add new columns to the DataFrame
    df['Retrieved Title'] = titles
    df['Paper ID'] = paper_ids
    df['Abstract'] = abstracts
    df['PDF Link'] = pdf_links
    df['embedding'] = embeddings

    # Save the results to a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(df, f)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    input_file = "graph based re-id.xlsx"
    output_file = "output_file.pkl"
    process_papers(input_file, output_file)