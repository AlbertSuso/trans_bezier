"""PROGRAM TO MAKE SILLY COMPROVATIONS"""

import torch
import numpy as np

batch_size = 10
temporal_size = 50

cov = torch.tensor([[3, 0], [0, 3]], dtype=torch.float32)
covariance = torch.empty((batch_size, temporal_size, 2, 2), dtype=torch.float32)
covariance[:, :] = cov
mean = torch. randint(0, 64, (batch_size, temporal_size, 2), dtype=torch.float32)
p = torch.empty((1, 64, 64, 1, 2), dtype=torch.float32)
for i in range(64):
    for j in range(64):
        p[0, i, j, 0, 0] = i
        p[0, i, j, 0, 1] = j

mean = mean.unsqueeze(1).unsqueeze(1)
cov_inv = torch.inverse(covariance).unsqueeze(1).unsqueeze(1)
divisor = torch.sqrt((2 * np.pi) ** 2 * torch.det(covariance)).unsqueeze(1).unsqueeze(1)

up = torch.matmul(p.unsqueeze(-2) - mean.unsqueeze(-2), cov_inv)
up = torch.matmul(up, p.unsqueeze(-1) - mean.unsqueeze(-1))
up = torch.exp(-0.5 * up)

output = up[:, :, :, :, 0, 0]/divisor
print("\n", output.shape)
