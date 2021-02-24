import torch


a = torch.tensor([True, True, False], dtype=torch.bool)
b = torch.tensor([True, False, False], dtype=torch.bool)

print(a+b)
