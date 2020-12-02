from torch.nn import CrossEntropyLoss
from torch import nn

class CBCELoss(nn.Module):
    def __init__(self) -> None:
        pass

    def forward(self, input, target):
        return 0


class GHMLoss(nn.Module):
    def __init__(self) -> None:
        pass

    def forward(self, input, target):
        return 0

class GGHMLoss(nn.Module):
    def __init__(self) -> None:
        pass

    def forward(self, input, target):
        return 0


def get_loss_fn(loss_fn):
    loss = eval(loss_fn)
    return loss.cuda()