from torch.nn import CrossEntropyLoss
from torch import nn
import numpy as np
import torch

class CBCELoss(nn.Module):
    def __init__(self, lt_factor, num_classes) -> None:
        super(CBCELoss).__init__()
        weight = np.linspace(1/lt_factor, 1, num=num_classes)
        self.ce_loss = CrossEntropyLoss(weight=torch.tensor(weight))

    def forward(self, input, target):
        return self.ce_loss(input, target)


class GHMLoss(nn.Module):
    # TODO complete GHMloss
    def __init__(self, num_classes) -> None:
        super(GHMLoss).__init__()
        pass

    def forward(self, input, target):
        return 0

class GroupGHMLoss(nn.Module):
    # TODO complete GGHMloss
    def __init__(self) -> None:
        super(GroupGHMLoss).__init__()
        pass

    def forward(self, input, target):
        return 0


def get_loss_fn(loss_fn, lt_factor, num_classes):
    if loss_fn == 'CrossEntropyLoss':
        return eval(loss_fn)().cuda()
    
    return 0