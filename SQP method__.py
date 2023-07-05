import numpy as np
from scipy.optimize import minimize

def objective(x):
    # Define the objective function to minimize
    return x[0]**2 + x[1]**2

def constraint(x):
    # Define the constraint function (example: x[0] + x[1] = 1)
    return x[0] + x[1] - 1

def jacobian(x):
    # Define the Jacobian matrix of the constraint function
    return np.array([1, 1])

def hessian(x, v):
    # Define the Hessian matrix of the Lagrangian function
    return np.array([[2, 0], [0, 2]])

def sqp_method(x0):
    x = x0
    epsilon = 1e-6
    max_iterations = 100
    
    v = 0  # Initialize Lagrange multipliers
    
    for iteration in range(max_iterations):
        # Solve the quadratic programming subproblem to obtain the search direction
        res = minimize(lambda x: objective(x) - v * constraint(x),
                       x,
                       jac=lambda x: 2 * x,
                       hess=lambda x: np.eye(len(x)) + np.outer(jacobian(x), jacobian(x)),
                       constraints={'type': 'eq', 'fun': constraint, 'jac': jacobian})
        dx = res.x - x
        
        # Update the Lagrange multipliers using the search direction
        if res.success and res.message == 'Optimization terminated successfully.':
            v = res.v
        else:
            v = 0
        
        # Check for convergence
        if np.linalg.norm(dx) < epsilon:
            break
        
        # Update the iterate
        x = res.x
    
    return x, objective(x)

# Example usage
x0 = np.array([0, 0])  # Initial guess
solution, min_value = sqp_method(x0)
print("Solution:", solution)
print("Minimum value:", min_value)


