# Import packages.
import cvxpy as cp
import numpy as np
# import pandas as pd

# Generate a random non-trivial linear program.
n = 9
s = 1
t = 9
r = 3

graph = [
    {'edge': '12', 'c': 1, 'a': 4},
    {'edge': '15', 'c': 2, 'a': 1},
    {'edge': '13', 'c': 3, 'a': 2},
    {'edge': '24', 'c': 2, 'a': 2},
    {'edge': '25', 'c': 3, 'a': 2},
    {'edge': '36', 'c': 2, 'a': 1},
    {'edge': '45', 'c': 0, 'a': 2},
    {'edge': '47', 'c': 5, 'a': 1},
    {'edge': '56', 'c': 1, 'a': 3},
    {'edge': '57', 'c': 1, 'a': 4},
    {'edge': '58', 'c': 0, 'a': 2},
    {'edge': '68', 'c': 2, 'a': 4},
    {'edge': '79', 'c': 1, 'a': 3},
    {'edge': '89', 'c': 3, 'a': 1},
]

graph_dict = {entry['edge']: entry for entry in graph}

A = np.array(
    [[graph_dict[f'{i}{j}']['a'] if f'{i}{j}' in graph_dict else 0
      for j in range(1, n + 1)]
     for i in range(1, n + 1)
     ])
C = np.array(
    [[graph_dict[f'{i}{j}']['c'] if f'{i}{j}' in graph_dict else 0
      for j in range(1, n + 1)]
     for i in range(1, n + 1)
     ])
print(A)
print('-'*20)
print(C)

R = cp.Variable((n, n), name='Transport_Amount', integer=True)

objective = cp.Minimize(cp.trace(C.T @ R))
constraints = [
    0 <= R,
    R <= A,
    cp.sum(R, axis=0) - cp.sum(R.T, axis=1) == 0,
    # cp.sum(R, axis=0)[s-1] == r,
    # cp.sum(R, axis=1)[t-1] == r,
]

prob = cp.Problem(objective,
                  constraints)
prob.solve()
# conflicting_constraints = []
# for constraint in constraints:
#     dual_val = constraint.dual_value
#     if dual_val != 0:
#         conflicting_constraints.append(constraint)
#
# # Print the conflicting constraints
# if conflicting_constraints:
#     print("Conflicting constraints:")
#     for constraint in conflicting_constraints:
#         print(constraint)
# else:
#     print("No conflicting constraints")

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution R")
print(R.value)
# print("A dual solution is")
# print(prob.constraints[0].dual_value)
