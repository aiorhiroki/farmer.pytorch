import torch
import matplotlib.pyplot as plt


class Logger:
    def __init__(self):
        self.logs = {}

    def set_progbar(self, nb_iters):
        self.prog_bar = ProgressBar(nb_iters)

    def get_progbar(self, loss, metrics):
        self.prog_bar.print_prog_bar(loss, metrics)

    def set_metrics(self, metric_names: list):
        for metric_name in metric_names:
            self.logs[metric_name] = []

    def get_latest_metrics(self):
        return self.prog_bar.get_latest_metrics()

    def update_metrics(self):
        self.logs['dice'] += [self.get_latest_metrics()]

    def plot_logs(self):
        for metric_name, history in self.logs.items():
            plt.plot(history)
            plt.savefig(f"{metric_name}.png")
            plt.close()


class ProgressBar:
    def __init__(self, nb_iters):
        self._nb_iters = nb_iters
        self._iter_i = 0
        self._total_loss = 0
        self._total_metrics = 0

    def print_prog_bar(self, loss, metrics, length=50):
        self._iter_i += 1
        prob_norm = length / self._nb_iters
        done = '=' * int(self._iter_i * prob_norm)
        todo = ' ' * int((self._nb_iters - self._iter_i) * prob_norm)
        cout = f"[{done + todo}] {self._iter_i}/{self._nb_iters}"

        self._total_loss += loss
        self._total_metrics += metrics
        cout += f" loss: {(self._total_loss / self._iter_i):.5g}"
        cout += f" dice: {(self._total_metrics / self._iter_i):.5g}"
        print("\r"+cout, end="")

    def get_latest_metrics(self):
        return self._total_metrics / self._iter_i


class SegMetrics:
    def __init__(self, class_weights=None):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.class_weights = class_weights

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
        threshold: float = None
    ):
        if threshold is not None:
            output = torch.where(output >= threshold, 1, 0)
            target = torch.where(target >= threshold, 1, 0)

        batch_size, num_classes, *dims = target.shape
        output = output.view(batch_size, num_classes, -1)
        target = target.view(batch_size, num_classes, -1)

        tp = (output * target).sum(2)
        fp = output.sum(2) - tp
        fn = target.sum(2) - tp
        tn = torch.prod(torch.tensor(dims)) - (tp + fp + fn)

        return tp, fp, fn, tn


class Fscore(SegMetrics):
    def __init__(self, class_weights=None):
        super().__init__(class_weights)

    def __call__(self, outputs, labels):
        return self._compute_metric(outputs, labels, self._fbeta_score)

    def _fbeta_score(self, tp, fp, fn, tn, beta=1):
        beta_tp = (1 + beta ** 2) * tp
        beta_fn = (beta ** 2) * fn
        score = beta_tp / (beta_tp + beta_fn + fp)
        return score
