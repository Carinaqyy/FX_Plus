import torch

class Test1(torch.nn.Module):
    def forward(self, x, y):
        lhs = torch.ops.aten.add(x, 1)
        rhs = torch.ops.aten.add(y, 2)
        return torch.ops.aten.add(lhs, rhs)
        