import fitz  # PyMuPDF
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import List, Dict, Union
import json
import hashlib
import re
import chromadb
from chromadb.utils import embedding_functions

class PDFRAGSystem:
    def __init__(self):
        
        self.client = chromadb.PersistentClient(path="../../chroma")
        self.embedding_func = embedding_functions.OpenAIEmbeddingFunction(
            api_key="API_KEY_IS_NOT_NEEDED",
            api_base="http://10.176.64.152:11435/v1",
            model_name="bge-m3"
        )

        # 创建/获取集合
        self.collection = self.client.get_or_create_collection(
            name="pdf_rag",
            embedding_function=self.embedding_fn
        )
        
        # 加载PDF配置
        with open("../dataset/paper.json") as f:
            self.pdf_config = json.load(f)
        
        # 文档分块参数
        self.chunk_size = 512  # 字符数
        self.overlap = 64     # 块间重叠字符数

    def _pdf_loader(self, path: str) -> List[str]:
        text = []
        with fitz.open(path) as doc:
            for page in doc:
                page_text = page.get_text()
                clean_text = re.sub(r'\s+', ' ', page_text).strip()
                text.append(clean_text)
        return text

    def _text_chunker(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.overlap
        return chunks

    def initialize_db(self):
        all_docs = []
        for item in self.pdf_config:
            pages = self._pdf_loader(item["path"])
            
            metadata = {
                "doc_id": str(item["id"]),
                "source": item["path"],
                "pages": len(pages)
            }

            base_id = hashlib.md5(item["path"].encode()).hexdigest()
            
            for page_num, page_text in enumerate(pages):
                chunks = self._text_chunker(page_text)
                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = f"{base_id}_p{page_num}_c{chunk_idx}"
                    all_docs.append({
                        "id": doc_id,
                        "text": chunk,
                        "metadata": {
                            **metadata,
                            "page": page_num + 1,
                            "chunk": chunk_idx + 1
                        }
                    })

        self.collection.add(
            documents=[d["text"] for d in all_docs],
            metadatas=[d["metadata"] for d in all_docs],
            ids=[d["id"] for d in all_docs]
        )

    def retrieve(
        self,
        query: str,
        top_n: int = 3,
        doc_filter: Union[List[int], None] = None
    ) -> List[Dict]:
        where_filter = {}
        if doc_filter:
            where_filter = {"doc_id": {"$in": [str(i) for i in doc_filter]}}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_n,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        return [{
            "text": doc,
            "score": 1 - distance,
            "metadata": meta
        } for doc, distance, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        )]
