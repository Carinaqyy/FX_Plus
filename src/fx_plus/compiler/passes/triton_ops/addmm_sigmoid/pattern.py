import torch


def pattern(a, b, c):
    mm = torch.ops.aten.mm(a, b)
    add = torch.ops.aten.add(mm, c)
    sigmoid = torch.ops.aten.sigmoid(add)
    return sigmoid