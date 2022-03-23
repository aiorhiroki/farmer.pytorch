import matplotlib.pyplot as plt
from .progress_bar import ProgressBar


class Logger:
    def __init__(self, result_dir):
        self.result_dir = result_dir
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
            plt.savefig(f"{self.result_dir}/{metric_name}.png")
            plt.close()
