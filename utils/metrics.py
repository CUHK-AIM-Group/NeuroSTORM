import torch
import numpy as np
from torch.nn import functional as F
from math import exp


class Metrics:
    @staticmethod
    def get_accuracy(y_hat, y):
        return (y_hat.argmax(dim=1) == y).float().mean()

    @staticmethod
    def get_accuracy_binary(y_hat, y):
        return ((y_hat >= 0) == y).float().mean()
