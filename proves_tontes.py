import torch
import numpy as np

from torch.optim import Adam


def f1(p, x):
    return 3*x**2 - 10*x + p

def f2(p, x):
    return 3*x**2 + p*x - 5

def f3(p, x):
    return p*x**1 -10*x - 5


best_loss = float('inf')

p = torch.rand(1, requires_grad=True)
x = np.pi

optimizer = Adam([p], lr=1e-3)

for i in range(1000000):
    val1 = f1(p, x)
    val2 = f2(p, x)
    val3 = f3(p, x)

    loss = val1 + 0.8*(val2 - val1) + 0.64*(val3 - val2)
    loss.backward()
    optimizer.step()
    p.grad.zero_()

    if i % 100000 == 99999:
        print("val1={}".format(val1))
        print("val2={}".format(val2))
        print("val3={}".format(val3))
        print("loss={}\n".format(loss))


