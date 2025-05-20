import numpy as np
from embedding import cosine_similarity
from bert_score import score as bert_score

def reference_evaluation(benchmark, test, metric='f1'):
    true_positives = len(set(test) & set(benchmark))
    pred_count = len(test)
    true_count = len(benchmark)
    
    precision = true_positives / pred_count if pred_count > 0 else 0.0
    recall = true_positives / true_count if true_count > 0 else 0.0
    
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
    _, _, F1 = bert_score(
        [test], 
        [benchmark],
        lang=lang,
        rescale_with_baseline=True,
        use_fast_tokenizer=True
    )
    return F1.item()

# Factory function to create evaluator using cosine similarity
def create_evaluator_cosine(ref_metric='f1'):
    return {
        "content": lambda benchmark, text, language: cosine_similarity(benchmark, text) * 10,
        "reference": lambda benchmark, text, language: reference_evaluation(benchmark, text, ref_metric) * 10
    }

# Factory function to create evaluator using bert_score
def create_evaluator_bert_score(lang='en', ref_metric='f1'):
    return {
        "content": lambda benchmark, text, language: bert_score_evaluation(benchmark, text, language) * 10,
        "reference": lambda benchmark, text, language: reference_evaluation(benchmark, text, ref_metric) * 10
    }
