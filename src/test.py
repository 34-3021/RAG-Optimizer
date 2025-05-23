from rag import PDFRAGSystem
from algorithm import algorithm_base

def main():
    rag = PDFRAGSystem()

    title = input("请输入论文标题：")
    quest = input("请输入问题：")
    language = input("请输入语言（Chinese/English）：")

    content, reference = algorithm_base(
        rag,
        question={
            "title": title,
            "quest": quest
        },
        language=language
    )

    print("生成的内容：", content)
    print("参考文献：", list(set(reference)))

if __name__ == "__main__":
    main()