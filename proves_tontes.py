import torch
import torch.nn as nn
import time

a = torch.randint(0, 2, (10,))
b = a < 1

print(a)
print(b)
print(torch.sum(b))

