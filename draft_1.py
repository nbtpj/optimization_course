# Import packages.
import numpy as np

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

R = np.abs(np.random.rand(n,n))
print(R)
objective = np.sum(C.T @ R)
constraints = [
    0 <= R,
    R <= A,
    np.sum(R, axis=0) - np.sum(R, axis=1) == 0,
    np.sum(R[0, :]) == r,
    np.sum(R[:, -1]) == r,
]
print(objective, constraints)

