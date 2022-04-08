import matplotlib.pyplot as plt
import json
from pathlib import Path
from .progress_bar import ProgressBar


class Logger:
    def __init__(self, result_dir):
        Path(result_dir).mkdir(exist_ok=True, parents=True)
        self.result_dir = result_dir
        self.dice_history = []

    def set_progbar(self, nb_iters):
        self.prog_bar = ProgressBar(nb_iters)

    def get_progbar(self, loss, dice=None, lr=None):
        self.prog_bar.print_prog_bar(loss, dice, lr)

    def get_latest_metrics(self):
        return self.prog_bar.get_latest_metrics()

    def update_metrics(self):
        self.dice_history += [self.get_latest_metrics()]

    def on_epoch_end(self):
        self._plot_logs()
        self._save_metric()

    def _plot_logs(self):
        plt.plot(self.dice_history)
        plt.savefig(f"{self.result_dir}/dice.png")
        plt.close()

    def _save_metric(self):
        scores = dict(dice=self.dice_history)
        with open(f"{self.result_dir}/scores.json", "w") as fw:
            json.dump(scores, fw)

        history_dict = {}
        history_dict['dice'] = []
        for epoch, history in enumerate(self.dice_history):
            history_dict['dice'] += [{'epoch': epoch, 'dice': history}]
        with open(f"{self.result_dir}/dice.json", "w") as fw:
            json.dump(history_dict, fw, indent=4)
