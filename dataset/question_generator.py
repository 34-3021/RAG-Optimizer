from PyPDF2 import PdfReader    
import re
import json
import requests
from tqdm import tqdm

def ask(prompt: str) -> str:
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
    return response.json()["choices"][0]["message"]["content"]

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    with open("paper.json") as f:
        papers = json.load(f)
    
    problems = []
    raw = []

    for paper in tqdm(papers):
        text = extract_text_from_pdf("../" + paper["path"])
        problem = ask(f"下面是一篇论文，请你仔细阅读，并提出两个通用性较强、且在脱离上下文之后能让读者理解的问题，例如不要问“这个模型使用了什么训练方法”，而是问“Falcon模型训练方法相较于其他模型的独特性”。两个问题需要一个是中文一个是英文，再提问的同时还需要提出一些要求，便于回答者解决问题，例如刚才那个问题可以提出“请你从模型结构、预训练过程、后训练过程的角度来回答”。问题与问题、问题与要求之间用换行符来分割。格式是：“中文问题：xxxxx\\n中文要求：xxxxx\\n英文问题：xxxxx\\n英文要求：xxxxx\\n”。请你严格按照指定格式输出。下面是论文内容{text}。").split("\n")
        raw.append(problem)
        problem = [i for i in problem if i != ""]
        question1 = (problem[0].strip() + "\n" + problem[1].strip()).replace("中文要求：", "").replace("中文问题：", "")
        question2 = (problem[2].strip() + "\n" + problem[3].strip()).replace("英文要求：", "").replace("英文问题：", "")
        title1 = ask(f"请你猜测下面这个问题所属的综述的标题是什么。请你只输出标题，不要输出任何提示性词语和其它内容，请你严格按照指定格式输出。。问题：{question1}")
        title2 = ask(f"请你猜测下面这个问题所属的综述的标题是什么。请你只输出标题，不要输出任何提示性词语和其它内容，请你严格按照指定格式输出。。问题：{question2}")
        problems.append({
            "title": title1,
            "quest": question1
        })
        problems.append({
            "title": title2,
            "quest": question2
        })

    with open("question_pre.json", "w", encoding="utf-8") as f:
        json.dump(problems, f, ensure_ascii=False, indent=4)
    
    with open("raw.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()