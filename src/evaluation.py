import numpy as np
from embedding import cosine_similarity
from bert_score import score as bert_score

def reference_evaluation(benchmark, test, metric='f1'):
    true_positives = len(set(test) & set(benchmark))
    pred_count = len(test)
    true_count = len(benchmark)
    
    precision = true_positives / pred_count if pred_count > 0 else 0.0
    recall = true_positives / true_count if true_count > 0 else 0.0

    print("precision:", precision)
    print("recall:", recall)
    
    if metric == 'precision':
        return precision
    elif metric == 'recall':
        return recall
    elif metric == 'f1':
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    else:
        raise ValueError("Invalid metric, choose from: precision/recall/f1")

def bert_score_evaluation(benchmark: str, test: str, lang: str) -> float:
    # print(benchmark, test, lang)
    _, _, F1 = bert_score([test], [benchmark], lang=lang)
    return F1.item()

# Factory function to create evaluator using cosine similarity
# [deprecated]
# def create_evaluator_cosine(ref_metric='f1'):
#     return {
#         "content": lambda benchmark, text, language: cosine_similarity(benchmark, text),
#         "reference": lambda benchmark, text: reference_evaluation(benchmark, text, ref_metric)
#     }

# Factory function to create evaluator using bert_score
def create_evaluator_bert_score(lang='en', ref_metric='f1'):
    return {
        "content": lambda benchmark, text, language: bert_score_evaluation(benchmark, text, language),
        "reference": lambda benchmark, text: reference_evaluation(benchmark, text, ref_metric)
    }
