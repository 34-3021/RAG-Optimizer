from rag import PDFRAGSystem
from algorithm import algorithm_base
from benchmark import test_score, parse_indicator, plot_distribution
from evaluation import create_evaluator_bert_score
import numpy as np
import json
import pickle

def main():
    rag = PDFRAGSystem()
    # rag.initialize_db()

    score = test_score(
        algorithm = algorithm_base,
        rag_searcher = rag,
        evaluator = create_evaluator_bert_score()
    )

    with open("result.json", "w") as f:
        json.dump(score, f)

if __name__ == "__main__":
    main()