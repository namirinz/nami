import torch
from typing import Union

class Accuracy:
    def __init__(self):
        self.count = 0
        self.total = 0

    def reset(self):
        self.count = 0
        self.total = 0

    def update(self, y_pred, y_true, mask=None):
        y_true = y_true.detach()
        y_pred = y_pred.detach()
        y_pred_idx = torch.argmax(y_pred, axis=-1)
        if mask != None:
            self.count += (torch.eq(y_true, y_pred_idx)* mask).sum()
        else:
            self.count += torch.eq(y_true, y_pred_idx).sum()
        self.total += len(y_true)
    
    def result(self):
        return self.count / self.total