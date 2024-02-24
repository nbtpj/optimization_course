import cvxpy as cp
import numpy as np

s = 1
t = 9
r = 3
n = 9

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
e = len(graph)

A = np.zeros((n, e))
b = np.zeros((n, 1))
c = np.zeros((e, 1))
a = np.zeros((e, 1))

for i, node in enumerate(range(1, n + 1)):
    for j, edge2 in enumerate(graph):
        if edge2['edge'][1:] == str(node):
            A[i, j] = -1  # in to a node
        if edge2['edge'][0] == str(node):
            A[i, j] = 1  # out from a node
    if node == s:
        b[i] = r  # only in
    elif node == t:
        b[i] = -r  # only out

for i, edge in enumerate(graph):
    c[i, 0] = edge['c']
    a[i, 0] = edge['a']

# -------------- Primal Problem ---------------

r = cp.Variable((e, 1), integer=True)

objective = cp.Minimize(c.T @ r)
constraints = [
    0 <= r,
    r <= a,
    A @ r == b,
]
prob = cp.Problem(objective, constraints)
prob.solve()

print("\nThe primal optimal value is", prob.value)
print('-' * 30)
for edge, u in zip(graph, r.value):
    if int(u[0]):
        print(f'edge: {edge["edge"]}, cost: {edge["c"]}, capacity: {edge["a"]}, '
              f'transport through: {int(u[0])}')

# -------------- Dual Problem ------------------

nu = cp.Variable((n, 1))
lamb = cp.Variable((e, 1))

dual_objective = cp.Maximize(-nu.T @ b - lamb.T @ a)
dual_constraints = [
    lamb >= 0,
    c.T + nu.T @ A + lamb.T >= 0,
]
prob_dual = cp.Problem(dual_objective, dual_constraints)
prob_dual.solve(solver=cp.ECOS)

print("\nThe dual optimal value is", prob_dual.value)
print('-' * 30)
