from PyPDF2 import PdfReader
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import List, Dict, Union
import json
import hashlib
import re
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

class PDFRAGSystem:
    def __init__(self):
        
        self.client = chromadb.PersistentClient(path="chroma")
        self.embedding_func = embedding_functions.OpenAIEmbeddingFunction(
            api_key="API_KEY_IS_NOT_NEEDED",
            api_base="http://10.176.64.152:11435/v1",
            model_name="bge-m3"
        )

        # 创建/获取集合
        self.collection = self.client.get_or_create_collection(
            name="pdf_rag",
            embedding_function=self.embedding_func
        )
        
        with open("dataset/paper.json") as f:
            self.pdf_config = json.load(f)
        
        # 文档分块参数
        self.chunk_size = 512  # 字符数
        self.overlap = 64     # 块间重叠字符数

    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _text_chunker(self, text: str) -> List[str]:
        return [
            text[start: start + self.chunk_size]
            for start in range(0, len(text), self.chunk_size - self.overlap)
        ]

    def initialize_db(self):

        for item in tqdm(self.pdf_config):
            all_docs = []
            # print(f"Processing {item['path']}")
            text = self.extract_text_from_pdf(item["path"])
            # print(f"Extracted {len(text)} characters from {item['path']}")
            
            metadata = {
                "doc_id": str(item["id"]),
                "source": item["path"]
            }

            base_id = hashlib.md5(item["path"].encode()).hexdigest()
            
            chunks = self._text_chunker(text)
            # print(f"Chunked {len(chunks)} chunks for {item['path']}")
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"{base_id}_c{chunk_idx}"
                all_docs.append({
                    "id": doc_id,
                    "text": chunk,
                    "metadata": metadata
                })

            self.collection.add(
                documents=[d["text"] for d in all_docs],
                metadatas=[d["metadata"] for d in all_docs],
                ids=[d["id"] for d in all_docs]
            )

    def retrieve(self, query: str, top_n: int = 3) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_n,
            include=["documents", "metadatas"]
        )

        return [{
            "text": doc,
            "metadata": meta
        } for doc, meta in zip(
            results["documents"][0],
            results["metadatas"][0]
        )]
