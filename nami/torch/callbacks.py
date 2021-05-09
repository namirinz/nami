import torch

class EarlyStopping:
    def __init__(self, patience: int, mode: str, min_delta: int = 0, restore_best_weights: bool = False, verbose: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        if mode == "min":
            self.best = 9999999
            self.op_fn = torch.less
        elif mode == "max":
            self.best = -1
            self.op_fn = torch.greater
        self.best_weights = None

    def check(self, monitor, model_state=None):
        monitor = monitor.detach()
        if self.op_fn(monitor, self.best):
            if self.verbose:
                print(f"monitor improved from {self.best:.6f} to {monitor:.6f}.")
            if self.restore_best_weights:
                self.best_weights = model_state
            self.best = monitor
            self.count = 0
        else:
            if self.verbose:
                print(f"monitor \"not\" improved from {self.best:.6f}.")
            self.count += 1
            if self.count == self.patience:
                if self.restore_best_weights:
                    return (True, self.best_weights)
                else: return (True)
        if self.restore_best_weights: return (False, self.best_weights)
        return (False)