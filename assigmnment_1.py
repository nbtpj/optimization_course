import cvxpy as cp
import numpy as np


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
d = np.zeros((1, n))
d[0, s-1] = -r
d[0, t-1] = r
h = np.ones((1, n))
R = cp.Variable((n, n), integer=True)

objective = cp.Minimize(cp.trace(C.T @ R))
constraints = [
    0 <= R,
    R <= A,
    h @ R - h @ R.T == d,
]
prob = cp.Problem(objective, constraints)
prob.solve()

print("\nThe optimal value is", prob.value)
print('-' * 30)
for i in range(1, n + 1):
    for j in range(1, n + 1):
        if R.value[i-1, j-1] != 0:
            print(f'edge: {i}{j}, cost: {C[i-1, j-1]}, capacity: {A[i-1, j-1]}, '
                  f'transport through: {int(R.value[i-1, j-1])}')
