import torch


def pattern(a, b, c, d, e):
    # (sigmoid(a @ b + c) - d) * (e * 0.0078125)
    mm = torch.ops.aten.mm(a, b)
    add_1 = torch.ops.aten.add(mm, c)
    sigmoid = torch.ops.aten.sigmoid(add_1)
    mul1 = torch.ops.aten.mul(d, -1)
    add_2 = torch.ops.aten.add(sigmoid, mul1)
    mul_2 = torch.ops.aten.mul(e, 0.0078125)
    mul = torch.ops.aten.mul(add_2, mul_2)
    return mul
    