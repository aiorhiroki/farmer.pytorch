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
