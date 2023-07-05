import numpy as np

def forward_backward_splitting(f_smooth, f_nonsmooth, prox_op, x0, alpha, max_iter):
    x = x0
    for i in range(max_iter):
        x_prev = x

        # Perform the forward step
        grad_smooth = smooth_gradient(x)
        forward_step = x - alpha * grad_smooth

        # Perform the backward step using the proximal operator
        x = prox_op(forward_step, alpha)

        # Check for convergence
        if np.linalg.norm(x - x_prev) < 1e-2:
            break

    return x

# Example usage
# Define the smooth function and its gradient
def smooth_function(x):
    return 0.5 * np.linalg.norm(x) ** 2

def smooth_gradient(x):
    return x

# Define the non-smooth function
def nonsmooth_function(x):
    return np.abs(x)

# Define the proximal operator for the non-smooth function
def prox_operator(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

# Set the initial point, step size, and maximum number of iterations
x0 = np.array([1, 2, 3])
alpha = 0.1
max_iter = 100

# Call the forward-backward splitting method
result = forward_backward_splitting(smooth_function, nonsmooth_function, prox_operator, x0, alpha, max_iter)

print("Optimal solution:", result)
