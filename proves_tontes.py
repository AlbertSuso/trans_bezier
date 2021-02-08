import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.randint(0, 2, (5, 6, 64, 2))

b = torch.randint(0, 5, (64,))

out = torch.empty((6, 64, 2))

out

print(a[b].shape)


