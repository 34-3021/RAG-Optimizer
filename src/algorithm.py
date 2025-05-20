import json
from typing import List, Dict, Tuple
import requests

def ask(prompt: str) -> str:
    response = requests.post(
        url = "http://10.176.64.152:11434/v1/chat/completions",
        json = {
            "model": "qwen2.5:7b",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def algorithm_base(
    rag,
    question: Dict[str, str],
    language: str
) -> Tuple[str, List[int]]:
    title = question["title"].strip()
    quest = question["quest"].strip()
    resp1 = ask(f"我正在写一篇和{title}相关的论文，要求是{quest}。与此同时，用户提供了一个论文库，你可以输出你需要的关键词，比如“强化学习”，请你尽可能地具体，便于系统搜索。你可以每行输出一个关键词，最多五行。请你只输出关键词，不要回复任何的其他内容。你必须用英文来回复关键词。")
    # print(resp1)
    keywords = resp1.split("\n")
    keywords = [k.strip() for k in keywords if k.strip()]

    related_papers = []
    for keyword in keywords:
        papers = rag.retrieve(
            query=keyword,
            top_n=5,
        )
        for paper in papers:
            resp2 = ask(f"我正在写一篇和{title}相关的论文，要求是{quest}。请你判断一下这一段论文是否和我的问题强相关，强相关请回复 Yes，不相关请回复 No。我希望你能在思考之后再给出答案，同时，请你将你的答案放在 \\bbox{{}}中，例如 \\bbox{{Yes}}。论文内容：{paper['text']}")
            if "\\bbox{Yes}" in resp2:
                related_papers.append(paper)
    reference = [int(paper["metadata"]["doc_id"]) for paper in related_papers]

    context = {}
    for paper in related_papers:
        if paper["metadata"]["doc_id"] not in context:
            context[paper["metadata"]["doc_id"]] = []
        context[paper["metadata"]["doc_id"]].append(paper["text"])

    context = "".join([f"论文ID：{k}，论文内容：{''.join(v)}\n" for k, v in context.items()])

    content = ask(f"我正在写一篇和{title}相关的论文，要求是{quest}，请你务必使用{'中文' if language == 'Chinese' else '英文'}编写。以下是相关的文献，请你在写文章的时候务必引用所有的文献，同时希望你能用方括号标注相关的文献，例如：“新时代就需要新发展[1]。”。文献：{context}")

    return (content, reference)
