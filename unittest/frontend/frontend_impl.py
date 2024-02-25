import torch

class Test1(torch.nn.Module):
    def forward(self, input):
        x = torch.ops.aten.detach(input)
        y = torch.ops.aten.add(x, 1)
        return y

class Test2(torch.nn.Module):
    def forward(self, x, y):
        z = torch.ops.aten.add.Tensor(x, y)
        return z
    
class Test3(torch.nn.Module):
    def forward(self, x):
        y = torch.ops.aten.view(x, [256, -1])
        return y
    
class Test4(torch.nn.Module):
    def forward(self, x, y):
        m = torch.ops.aten.add.Tensor(x, y)
        n = torch.ops.aten.add.Tensor(x, y)
        c = torch.ops.aten.add(m, 1)
        return torch.ops.aten.add(c, n)
