import torch
import torch.nn as nn
import time

a = torch.randint(0, 5, (3, 64, 1, 1))
b = torch.randint(0, 5, (3, 64, 2, 2))

c = a*b

for i in range(3):
    for j in range(64):
        for k in range(2):
            for s in range(2):
                assert c[i, j, k, s] == a[i, j, 0, 0] * b[i, j, k, s]

print(c.shape)
