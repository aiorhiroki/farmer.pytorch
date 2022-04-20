import torch


class SegMetrics:
    def __init__(self):
        self.total_confusion = None

    @torch.no_grad()
    def calc_confusion(
            self, output: torch.LongTensor, target: torch.LongTensor,
            from_logits=False, threshold=None):
        if from_logits:
            output = torch.nn.functional.logsigmoid(output).exp()
        if threshold is not None:
            output = torch.where(output >= threshold, 1, 0)
            target = torch.where(target >= threshold, 1, 0)

        batch_size, num_classes, *dims = target.shape
        output = output.view(batch_size, num_classes, -1)
        target = target.view(batch_size, num_classes, -1)

        tp = (output * target).sum(2).sum(0, keepdim=True)
        fp = output.sum(2).sum(0, keepdim=True) - tp
        fn = target.sum(2).sum(0, keepdim=True) - tp
        return torch.cat((tp, fp, fn))

    def compute_metric(self, confusion, metric_fn, class_weights=1.0):
        if self.total_confusion is None:
            self.total_confusion = confusion
        else:
            self.total_confusion += confusion
        tp, fp, fn = self.total_confusion
        class_weights = torch.tensor(class_weights).to(tp.device)
        class_weights = class_weights / class_weights.sum()
        score = metric_fn(tp, fp, fn)
        score = self._handle_zero_division(score)
        score = (score * class_weights).mean()
        return score

    def _handle_zero_division(self, x):
        nans = torch.isnan(x)
        value = torch.tensor(0, dtype=x.dtype).to(x.device)
        x = torch.where(nans, value, x)
        return x

    @staticmethod
    def dice(tp, fp, fn, beta=1):
        beta_tp = (1 + beta ** 2) * tp
        beta_fn = (beta ** 2) * fn
        score = beta_tp / (beta_tp + beta_fn + fp)
        return score
