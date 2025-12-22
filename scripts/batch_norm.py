import numpy as np

# Constraints
# 1 ≤ N ≤ 10,000
# 1 ≤ C ≤ 1,024
# eps = 1e-5
# -100.0 ≤ input values ≤ 100.0
# 0.1 ≤ gamma values ≤ 10.0
# -10.0 ≤ beta values ≤ 10.0


def batch_norm(x, gamma, beta, eps):
    mu = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_hat = (x - mu) / np.sqrt(var + eps)
    y = gamma * x_hat + beta
    return y


input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
gamma = np.array([1.0, 1.0])
beta = np.array([0.0, 0.0])
eps = 1e-5
expected_output = np.array([[-1.224, -1.224], [0.0, 0.0], [1.224, 1.224]])

output = batch_norm(input, gamma, beta, eps)

print(np.allclose(expected_output, output, rtol=1e-3))
