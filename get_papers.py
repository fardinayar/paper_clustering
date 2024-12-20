import pandas as pd
from sem_scholar_api import get_paper
import time
import json
import os

def process_papers(input_file, output_file):
    """
    Process paper information from an input Excel file and save the results to a JSON file.

    This function reads paper titles from an input Excel file, retrieves additional
    information for each paper using the get_paper function, and saves the results
    to a JSON file with added fields for retrieved title, paper ID, abstract,
    PDF link, and embedding. It saves progress after each paper and can resume from where it left off.

    Parameters:
    input_file (str): The path to the input Excel file containing paper titles.
    output_file (str): The path where the output JSON file will be saved.

    Returns:
    None

    Note:
    The function prints progress messages and the final save location to the console.
    It also includes a 2-second delay between processing each paper to avoid hitting rate limits.
    """
    # Read the input Excel file
    df = pd.read_excel(input_file)

    # Load existing results if the output file exists
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        processed_titles = set(result['Original Title'] for result in results)
    else:
        results = []
        processed_titles = set()

    # Process each paper title
    for idx, title in enumerate(df['Paper Title']):
        if title in processed_titles:
            print(f"#{idx} Skipping already processed: {title}")
            continue

        print(f"#{idx} Processing: {title}")
        paper_info = get_paper(title)

        result = {
            'Original Title': title,
            'Retrieved Title': 'N/A',
            'Paper ID': 'N/A',
            'Abstract': 'N/A',
            'PDF Link': 'N/A',
            'Embedding': 'N/A'
        }

        if paper_info:
            result['Retrieved Title'] = title
            result['Paper ID'] = paper_info.get('paperId', 'N/A')
            result['Abstract'] = paper_info.get('abstract', 'N/A')
            result['Embedding'] = paper_info.get('embedding', 'N/A')
            if paper_info.get('openAccessPdf'):
                result['PDF Link'] = paper_info['openAccessPdf'].get('url', 'N/A')

        results.append(result)
        processed_titles.add(title)

        # Save the results to a JSON file after each paper
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"Progress saved to {output_file}")

        # Add a small delay to avoid hitting rate limits
        time.sleep(2)

    print(f"All results saved to {output_file}")

if __name__ == "__main__":
    input_file = "graph based re-id.xlsx"
    output_file = "output_file.json"
    process_papers(input_file, output_file)