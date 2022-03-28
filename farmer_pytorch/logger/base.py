import matplotlib.pyplot as plt
import json
from pathlib import Path
from .progress_bar import ProgressBar


class Logger:
    def __init__(self, result_dir):
        Path(result_dir).mkdir(exist_ok=True, parents=True)
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

    def on_epoch_end(self):
        self._plot_logs()
        self._save_metric()

    def _plot_logs(self):
        for metric_name, history in self.logs.items():
            plt.plot(history)
            plt.savefig(f"{self.result_dir}/{metric_name}.png")
            plt.close()

    def _save_metric(self):
        scores = {m: history[-1] for m, history in self.logs.items()}
        with open(f"{self.result_dir}/scores.json", "w") as fw:
            json.dump(scores, fw)

        history_dict = {}
        for m, historys in self.logs.items():
            history_dict[m] = []
            for epoch, history in enumerate(historys):
                history_dict[m] += [{'epoch': epoch, m: history}]
            with open(f"{self.result_dir}/{m}.json", "w") as fw:
                json.dump(history_dict, fw, indent=4)
