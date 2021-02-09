import torch

def foo(tensor):
    tensor = tensor+1
    return tensor

a = torch.ones(5)

b = foo(a)

print(a)
print(b)



