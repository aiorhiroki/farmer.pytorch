

class GetPredictionABC:
    batch_size: int
    gpu: int

    def __init__(self, model, metrics_func, val_data):
        self.model = model
        self.metrics_func = metrics_func
        self.val_data = val_data
