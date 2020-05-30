import torch



def rbf(x,c,h):
    return torch.exp(-1.0/h * (x - c) * (x - c))


x = torch.tensor([2.], requires_grad=True)
print(x.requires_grad)

k = rbf(x,0,1)
print(k)
k.backward()
print(x.grad)
print(x)
x = x + x.grad
print(x)
