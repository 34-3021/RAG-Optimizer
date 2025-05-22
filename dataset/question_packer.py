import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\src")
from rag import PDFRAGSystem
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import requests

def ask(prompt: str) -> str:
    # print(f"asking...{prompt}")
    response = requests.post(
        url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        json = {
            "model": "deepseek-v3-250324",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "stream": False,
        },
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer c1a90a6a-196e-4aff-87e5-0f5ac5dbaad7"
        }
    )
    response.raise_for_status()
    # print(f"response...{response.json()['choices'][0]['message']['content']}")
    return response.json()["choices"][0]["message"]["content"]

def algorithm_base(rag, question: Dict[str, str]) -> Tuple[str, List[int]]:
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
        # print(f"papers...{papers}")
        for paper in papers:
            resp2 = ask(f"我正在写一篇和{title}相关的论文，要求是{quest}。请你判断一下这一段论文是否和我的问题相关性较高，相关性较高请回复 Yes，不相关请回复 No。我希望你能在思考之后再给出答案，同时，请你将你的答案放在 \\bbox{{}}中，例如 \\bbox{{Yes}}。论文内容：{paper['text']}")
            if "\\bbox{Yes}" in resp2:
                related_papers.append(paper)
    reference = [int(paper["metadata"]["doc_id"]) for paper in related_papers]

    reference = list(set(reference))

    # print(f"reference...{reference}")

    context = {}
    for paper in related_papers:
        if paper["metadata"]["doc_id"] not in context:
            context[paper["metadata"]["doc_id"]] = []
        context[paper["metadata"]["doc_id"]].append(paper["text"])

    context = "".join([f"论文ID：{k}，论文内容：{''.join(v)}\n" for k, v in context.items()])

    chinese_content = ask(f"我正在写一篇和{title}相关的论文，要求是{quest}，请你务必使用中文编写。请你不要输出任何和论文内容无关的文字和提示语，也不要使用 Markdown 语法。以下是相关的文献，请你在写文章的时候务必引用所有的文献，同时希望你能用方括号标注相关的文献，例如：“新时代就需要新发展[1]。”。文献：{context}")
    english_content = ask(f"我正在写一篇和{title}相关的论文，要求是{quest}，请你务必使用英文编写。请你不要输出任何和论文内容无关的文字和提示语，也不要使用 Markdown 语法。以下是相关的文献，请你在写文章的时候务必引用所有的文献，同时希望你能用方括号标注相关的文献，例如：“新时代就需要新发展[1]。”。文献：{context}")
    return (chinese_content, english_content, reference)

def main():
    rag = PDFRAGSystem(path="../chroma")
    with open("question_pre.json", "r", encoding='utf-8') as f:
        questions = json.load(f)

    for question in tqdm(questions[38:]):
    # for question in questions[:2]:
        # print(question["title"], question["quest"])
        chinese_content, english_content, reference = algorithm_base(rag, {
            "title": question["title"],
            "quest": question["quest"]
        })
        # print(chinese_content, english_content, reference)

        with open("question.json", "a", encoding="utf-8") as f:
            json.dump({
                "Q": [[question["title"], question["quest"] + "使用中文编写。"], [question["title"], question["quest"] + "使用英文编写。"]],
                "A": [chinese_content, english_content],
                "R": [reference, reference]
            }, f, ensure_ascii=False, indent=4)
            f.write("\n\n\n")

if __name__ == "__main__":
    main()