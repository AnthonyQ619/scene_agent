import numpy as np
import os
os.add_dll_directory(r"C:\\Users\\Anthony\\Desktop\\VCPKG\\vcpkg\\installed\\x64-windows\\bin")
os.add_dll_directory(r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin")
os.add_dll_directory(r"C:\\Program Files\\NVIDIA cuDSS\\v0.7\\bin\\12")
import pyceres

# Simple cost: residual = x - 5
class QuadraticCost(pyceres.CostFunction):
    def __init__(self):
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(self, parameters, residuals, jacobians):
        x_val = parameters[0][0]
        residuals[0] = x_val - 5.0

        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][0] = 1.0
        return True

x = np.array([0.0], dtype=float)

problem = pyceres.Problem()
cost = QuadraticCost()
problem.add_residual_block(cost, None, [x])

opts = pyceres.SolverOptions()
opts.dense_linear_algebra_library_type = pyceres.DenseLinearAlgebraLibraryType.CUDA
opts.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
opts.minimizer_progress_to_stdout = True

# Must create a SolverSummary to pass in
summary = pyceres.SolverSummary()

# Solve (API requires summary passed in)
pyceres.solve(opts, problem, summary)

print("Summary attributes:", dir(summary))

# Print summary
print("Final cost:", summary.final_cost)
if hasattr(summary, "message"):
    print("Status message:", summary.message)

print("Linear solver used:", summary.linear_solver_type_used)
print("Dense LA library used:", summary.dense_linear_algebra_library_type)
print("Optimized x:", x[0])