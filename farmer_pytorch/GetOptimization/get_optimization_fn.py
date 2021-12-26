import matplotlib.pyplot as plt


# プログレスバーの作成
def get_prog_bar(iter_i, iter_total, length=50):
    prob_norm = length / iter_total
    done = '=' * int(iter_i * prob_norm)
    todo = ' ' * int((iter_total - iter_i) * prob_norm)
    prob_bar = done+todo
    return f"[{prob_bar}] {iter_i}/{iter_total}"


# metricsのプロット
def plot_metrics(logs):
    for metric_name, history in logs.items():
        plt.plot(history)
        plt.savefig(f"{metric_name}.png")
        plt.close()
