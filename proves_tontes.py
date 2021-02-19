import torch
import torch.nn as nn
import torch.nn.functional as F


a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(a.shape)

probs = F.softmax(a, dim=0)
print(probs)

probs = F.softmax(a, dim=1)
print(probs)

