import torch
from itertools import permutations

perm = permutations(range(3))
perm = torch.tensor([p for p in perm], dtype=torch.long)

perm_tokens = torch.empty((perm.shape[0], 12))
for j in range(perm.shape[1]):
    for k in range(3+1):
        perm_tokens[:, j*(3+1) + k] = (3+1)*perm[:, j] + k

print(perm_tokens, "\n")

for n in range(perm.shape[1]):
    perm_tokens[:, n*(3+1):n*(3+1)+3] = perm_tokens[:, n*(3+1):n*(3+1)+3].flip(1)

print(perm_tokens)



