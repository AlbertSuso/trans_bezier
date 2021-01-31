import torch


a = torch.tensor([1, 2], dtype=torch.float32, requires_grad=True)
b = torch.tensor([1, 2], dtype=torch.float32, requires_grad=True)
c = 3*a
d = 5*b
loss_a = torch.sum(c)
loss_b = torch.sum(d)
loss_a.backward()
loss_b.backward()
print(a.grad)
print(b.grad)



a = torch.tensor([1, 2], dtype=torch.float32, requires_grad=True)
b = torch.tensor([1, 2], dtype=torch.float32, requires_grad=True)
c = 3*a
d = 5*b

c[0] = d[0]

loss_a = torch.sum(c)
loss_a.backward()
print(a.grad)
print(b.grad)