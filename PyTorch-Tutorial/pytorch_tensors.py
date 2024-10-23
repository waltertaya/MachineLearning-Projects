import torch
import numpy as np
''' As in numpy array, the arrays representing a matrix or a vector,
 also tensors represents a matrix or vector.
'''

x = torch.empty(3)
#  2D tensors
y = torch.empty(2, 3)
# 1D, 2D, 3D and so on

# Instead of empty tensors, we can also initialize the tensors
# e.g .zeros, ones
a = torch.ones(2, 1)
b = torch.zeros(3, 1)

# Also we can assign the tensors random values by using the .rand in the torch
matrix = torch.rand(4, 4, 4)

# Specifying the datatype of the values in the tensor

matrix = torch.ones(2, 2, dtype=torch.int64)

# print(x)
# print(y)
# print(a)
# print(matrix.dtype)

# Converting an numpy array / python list to tensor

x = torch.tensor([34, 23, 56])
# Check the size of the tensor
print(x.size())

x = torch.rand(4, 4)
y = torch.rand(4, 4)

# Basic operation to the tensors
z = x + y
z = torch.add(x, y)

z = x - y
z = torch.sub(x, y)

z = x * y
z = torch.mul(x, y)

z = x / y
z = torch.div(x, y)

# in_place operations => meaning replaces the existing values
# N/B: Most of the torch methods ending with _ are mostly in place
x.add_(y)
x.sub_(y)
z.mul_(y)
y.div_(x)

# Slicing the tensor

c = torch.rand(5, 6)
# print(c)
# print(c[:, 1])
# print(c[2, :])
# # If you have only one tensor, you can extract the value
# print(c[2, 1].item())

# Reshaping the tensor using the .view method in the torch
d = c.view(30) # N/B: Reshaping to one 1D, the tensor should have the same number of elements
# Reshaping to 2D, we supply -1 and number of columns to the view method
e = c.view(-1, 10)

# Converting the tensor to numpy array and vice versa
a = torch.ones(6)
b = a.numpy()

x = np.ones(7)
y = torch.from_numpy(x)
