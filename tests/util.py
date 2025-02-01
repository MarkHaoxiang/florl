import torch
from torch import nn


def initialise_parameters_to_float(model: nn.Module, val: float = 0.0):
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(val)
