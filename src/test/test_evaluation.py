import unittest
from unittest.mock import patch
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import reference_evaluation, bert_score_evaluation

class TestEvaluationFunctions(unittest.TestCase):

    # Tests for reference_evaluation
    def test_perfect_match_f1(self):
        benchmark = [1, 2, 3]
        test = [1, 2, 3]
        self.assertAlmostEqual(reference_evaluation(benchmark, test, 'f1'), 1.0)

    def test_partial_overlap_precision(self):
        benchmark = [1, 2, 3]
        test = [1, 2]
        self.assertAlmostEqual(reference_evaluation(benchmark, test, 'precision'), 1.0)

    def test_no_overlap_recall(self):
        benchmark = [1, 2]
        test = [3, 4]
        self.assertAlmostEqual(reference_evaluation(benchmark, test, 'recall'), 0.0)

    def test_empty_benchmark(self):
        benchmark = []
        test = [1, 2]
        self.assertAlmostEqual(reference_evaluation(benchmark, test, 'f1'), 0.0)

    def test_empty_test(self):
        benchmark = [1]
        test = []
        self.assertAlmostEqual(reference_evaluation(benchmark, test, 'precision'), 0.0)

    def test_both_empty(self):
        benchmark = []
        test = []
        self.assertAlmostEqual(reference_evaluation(benchmark, test, 'f1'), 0.0)

    # Tests for bert_score_evaluation with mocking
    @patch('bert_score.score')
    def test_bert_score_same_text(self, mock_score):
        mock_score.return_value = (None, None, np.array([0.95]))
        result = bert_score_evaluation("hello", "hello", "en")
        self.assertAlmostEqual(result, 0.95, places=2)

    @patch('bert_score.score')
    def test_bert_score_different_text(self, mock_score):
        mock_score.return_value = (None, None, np.array([0.3]))
        result = bert_score_evaluation("hello", "world", "en")
        self.assertAlmostEqual(result, 0.3, places=2)

    @patch('bert_score.score')
    def test_bert_score_language_handling(self, mock_score):
        mock_score.return_value = (None, None, np.array([0.8]))
        result = bert_score_evaluation("测试", "测试", "zh")
        mock_score.assert_called_with(
            ["测试"], ["测试"], lang="zh",
            rescale_with_baseline=True,
            use_fast_tokenizer=True
        )
        self.assertAlmostEqual(result, 0.8, places=2)

if __name__ == '__main__':
    unittest.main()
