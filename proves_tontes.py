import torch
import torch.nn as nn

batch_size = 5

out = torch.empty((1, 5, 512))
a = torch.zeros((0, batch_size, 2), dtype=torch.float32)

layer = nn.Linear(2, 512)

out[50:] = 23
print(out)
