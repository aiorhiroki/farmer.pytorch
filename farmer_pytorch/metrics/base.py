import torch


class SegMetrics:
    def __init__(self, from_logits=True, threshold=None, class_weights=None):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.from_logits = from_logits
        self.threshold = threshold
        self.class_weights = class_weights

    def reset_state(self):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None

    def _compute_metric(self, outputs, labels, metric_fn):
        self.get_confusion(outputs, labels)
        class_weights = self.class_weights or 1.0
        class_weights = torch.tensor(class_weights).to(self.tp.device)
        class_weights = class_weights / class_weights.sum()
        tp = self.tp.sum(0)
        fp = self.fp.sum(0)
        fn = self.fn.sum(0)
        tn = self.tn.sum(0)
        score = metric_fn(tp, fp, fn, tn)
        score = self._handle_zero_division(score)
        score = (score * class_weights)
        return score

    def _handle_zero_division(self, x):
        nans = torch.isnan(x)
        value = torch.tensor(0, dtype=x.dtype).to(x.device)
        x = torch.where(nans, value, x)
        return x

    def get_confusion(self, outputs, labels):
        tp, fp, fn, tn = self.get_stats_multilabel(outputs, labels)
        if self.tp is None:
            self.tp, self.fp, self.fn, self.tn = tp, fp, fn, tn
        else:
            self.tp = torch.cat((self.tp, tp))
            self.fp = torch.cat((self.fp, fp))
            self.fn = torch.cat((self.fn, fn))
            self.tn = torch.cat((self.tn, tn))

    @torch.no_grad()
    def get_stats_multilabel(
        self,
        output: torch.LongTensor,
        target: torch.LongTensor,
    ):
        if self.from_logits:
            output = torch.nn.functional.logsigmoid(output).exp()
        if self.threshold is not None:
            output = torch.where(output >= self.threshold, 1, 0)
            target = torch.where(target >= self.threshold, 1, 0)

        batch_size, num_classes, *dims = target.shape
        output = output.view(batch_size, num_classes, -1)
        target = target.view(batch_size, num_classes, -1)

        tp = (output * target).sum(2)
        fp = output.sum(2) - tp
        fn = target.sum(2) - tp
        tn = torch.prod(torch.tensor(dims)) - (tp + fp + fn)

        return tp, fp, fn, tn
