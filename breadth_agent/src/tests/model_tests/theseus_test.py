import torch
import theseus as th

# Create a variable with 3-D random data of batch size = 2 and name "x"
x = th.Variable(torch.randn(2, 3), name="x")
print(f"x: Named variable with 3-D data of batch size 2:\n  {x}\n")

# Create an unnamed variable. A default name will be created for it
y = th.Variable(torch.zeros(1, 1))
print(f"y: Un-named variable:\n  {y}\n")

# Create a named SE2 (2D rigid transformation) specifying data (batch_size=2)
z = th.SE2(x_y_theta=torch.zeros(2, 3).double(), name="se2_1")
print(f"z: Named SE2 variable:\n  {z}")