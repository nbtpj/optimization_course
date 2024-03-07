import sccf
import cvxpy as cp
import numpy as np

# ===== syntax test ======
# x = cp.Variable(10)
# objective = sccf.minimum(cp.sum_squares(x), 1.0) + sccf.minimum(cp.sum_squares(x - .1), 1.0)
# constraints = [cp.sum(x) == 1.0, x >= 0, x <= 1]
# prob = sccf.Problem(objective, constraints)
# print(prob.solve())

# ===== syntax test2 ======
# N = 10
# x = cp.Variable(N)
# y = cp.Variable(N)
# c = cp.Variable(1)
# M = cp.Variable(N)
# f_is = [sccf.MinExpression((M[t] - y[t]) ** 2, 0.5) for t in range(N)]
# objective = sccf.SumOfMinExpressions(f_is)
# constraints = [cp.sum(x) == 1.0, x >= 0, x <= 1, y == 2, M == x + c]
# prob = sccf.Problem(objective, constraints)
# print(prob.solve())

# ===== exp1 =============
N = 20
true_theta = 1
x = np.random.normal(0, 1, size=(N,))
z = np.random.normal(0, 1, size=(N,))
y = true_theta * x  # this is true function we need to estimate of linear regression
noisy_y = y + 0.1 * z  # make it noisy
indices = np.arange(N)
np.random.shuffle(indices)
y_with_outliers = np.copy(y)
y_with_outliers[indices[:5]] *= -1

theta = cp.Variable(1)
x_train = cp.Parameter(N, value=x)
y_train = cp.Parameter(N, value=y_with_outliers)

f_is = [sccf.MinExpression(cp.square(x_train[t] * theta[0] - y_train[t]), 0.2)
        for t in range(N)]
objective = sccf.SumOfMinExpressions(f_is) + 0.1 * theta[0]**2
prob = sccf.Problem(objective)
prob.solve()

import matplotlib.pyplot as plt
import numpy as np

plt.scatter(x, y_with_outliers)

# sample x
sample_x = np.linspace(np.amin(x), np.amax(x), 30)
predicted_y = sample_x * theta.value
plt.plot(sample_x, predicted_y)

plt.show()
print(x, y_with_outliers)
# x = [ 0.17297869  0.3331957   0.01077415 -0.96149778  1.24155836  0.60004288
#   0.11426303  1.04837858  0.87818471 -1.00521324  0.64116968 -0.1811937
#   0.40110978  0.28536152  0.6390865  -1.8414436  -0.16605205  0.67136218
#  -0.56843577 -1.45858292]
# y_with_outliers = [ 0.17297869 -0.3331957   0.01077415 -0.96149778  1.24155836  0.60004288
#  -0.11426303  1.04837858  0.87818471 -1.00521324  0.64116968 -0.1811937
#  -0.40110978  0.28536152  0.6390865  -1.8414436   0.16605205  0.67136218
#   0.56843577 -1.45858292]
