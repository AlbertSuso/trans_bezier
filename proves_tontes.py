import torch
import numpy as np

from torch.optim import Adam

def f(p, x):
    return 3*x**2 + p*x - 5


best_loss = float('inf')

p = torch.rand(1, requires_grad=True)
x = np.pi

optimizer = Adam([p], lr=1e-3)

for i in range(100000):
    val = f(p, x)

    loss = val - val.detach()
    loss.backward()
    optimizer.step()
    p.grad.zero_()

    if i % 10000 == 9999:
        print("val={}".format(val))
        print("loss={}\n".format(loss))


