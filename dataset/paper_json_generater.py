# find all the .pdf files in directore ./paper/ and write a json file
# format:
# [{ "id": 1, "path": "paper/1.pdf" }, ...]
# and save it to ./paper.json

import os
import json
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

def find_pdfs_in_directory(directory: str) -> List[Dict[str, str]]:
    """
    Find all PDF files in the given directory and return a list of dictionaries
    containing the file paths.

    Args:
        directory (str): The directory to search for PDF files.

    Returns:
        List[Dict[str, str]]: A list of dictionaries with file paths.
    """
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in tqdm(files):
            if file.endswith('.pdf'):
                pdf_files.append({"id": len(pdf_files) + 1, "path": os.path.join(root, file)})
    return pdf_files

def save_to_json(data: List[Dict[str, str]], output_file: str) -> None:
    """
    Save the given data to a JSON file.

    Args:
        data (List[Dict[str, str]]): The data to save.
        output_file (str): The path to the output JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # Define the directory to search for PDF files
    directory = './paper/'
    
    # Find all PDF files in the directory
    pdf_files = find_pdfs_in_directory(directory)
    
    # Save the list of PDF files to a JSON file
    output_file = './paper.json'
    save_to_json(pdf_files, output_file)
    
    print(f"Saved {len(pdf_files)} PDF files to {output_file}")