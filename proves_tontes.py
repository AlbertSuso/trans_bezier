import torch

a = torch.ones((10, 2), dtype=torch.long)

idx = torch.tensor([0, 2, 4, 6, 8], dtype=torch.long)
vals = torch.tensor([[-5, -32]], dtype=torch.long)

a[idx] = vals
print(a)