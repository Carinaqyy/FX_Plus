import torch

def pattern(a, b, c, d):
    mm = torch.ops.aten.mm(a, b)
    mul1 = torch.ops.aten.mul(mm, c)
    mul2 = torch.ops.aten.mul(mul1, 1.0)
    ne = torch.ops.aten.ne(d, 0)
    mul3 = torch.ops.aten.mul(mul2, ne)
    return mul3