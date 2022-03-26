from . import base


class Dice(base.SegMetrics):
    def __init__(self, from_logits=True, threshold=None, class_weights=None):
        super().__init__(from_logits, threshold, class_weights)

    def __call__(self, outputs, labels):
        return self._compute_metric(outputs, labels, self._fbeta_score)

    def _fbeta_score(self, tp, fp, fn, tn, beta=1):
        beta_tp = (1 + beta ** 2) * tp
        beta_fn = (beta ** 2) * fn
        score = beta_tp / (beta_tp + beta_fn + fp)
        return score
