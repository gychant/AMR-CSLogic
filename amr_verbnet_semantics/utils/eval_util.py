"""
Utility functions for evaluation
"""


class Metric(object):
    def __init__(self):
        self.cnt_samples = 0
        self.cnt_samples_wo_true_triples = 0
        self.cnt_samples_wo_pred_triples = 0
        self.sum_prec = 0
        self.sum_recall = 0
        self.sum_f1 = 0

