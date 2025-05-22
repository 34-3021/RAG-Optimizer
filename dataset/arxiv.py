import os
import requests
from datetime import datetime, timedelta
from arxiv import Client, Search, SortCriterion, SortOrder
import openai

def fetch_papers():
    search_query = "(all:\"machine learning systems\" OR all:MLSys OR all:\"distributed machine learning\" OR all:\"model serving\") AND (cat:cs.LG OR cat:cs.DC)"
    client = Client()
    search = Search(
        query=search_query,
        sort_by=SortCriterion.SubmittedDate,
        sort_order=SortOrder.Descending,
        max_results=100
    )
    
    papers = []
    cutoff_date = datetime.now() - timedelta(days=5)
    
    for result in client.results(search):
        if result.published.replace(tzinfo=None) > cutoff_date:
            papers.append({
                "title": result.title,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "abstract": result.summary,
                "authors": [a.name for a in result.authors],
                "published": result.published
            })
    print(f"Found {len(papers)} papers.")
    return papers

def download_pdf(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
