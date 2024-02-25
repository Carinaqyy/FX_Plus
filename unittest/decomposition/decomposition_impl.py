import torch

class TestAddmm(torch.nn.Module):
    def forward(self, bias, lhs, rhs):
        return torch.ops.aten.addmm(bias, lhs, rhs)