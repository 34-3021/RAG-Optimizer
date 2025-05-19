import numpy as np
from services.embedding import EmbeddingService

def content_similarity(benchmark, test):
    """内容相似度评估（余弦相似度）"""
    embeds = EmbeddingService.get_embeddings([benchmark, test])
    vec_a = np.array(embeds.data[0].embedding)
    vec_b = np.array(embeds.data[1].embedding)
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

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

# Factory function to create evaluator
def create_evaluator(ref_metric='f1'):
    return {
        "content": lambda b, t: content_similarity(b, t) * 10,
        "reference": lambda b, t: reference_evaluation(b, t, ref_metric) * 10
    }
