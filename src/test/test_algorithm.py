import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm import ask, algorithm_base

def test_ask():
    print(ask("Hello, how do you do?"))
    print(ask("你好，你好吗？"))

def test_algorithm_base():
    algorithm_base(
        rag=None,
        question={
            "title": "A Survey on Reinforcement Learning",
            "quest": "What are the main challenges in reinforcement learning?"
        },
        language="English"
    )

if __name__ == "__main__":
    test_ask()
    test_algorithm_base()