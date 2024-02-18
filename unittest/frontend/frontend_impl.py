import torch

class Test1(torch.nn.Module):
    def forward(self, input):
        x = torch.ops.aten.detach(input)
        y = torch.ops.aten.add(x, 1)
        return y

class Test2(torch.nn.Module):
    def forward(self, input):
        x = torch.ops.aten.detach(input)
        y = torch.ops.aten.add(x, 1)
        return y