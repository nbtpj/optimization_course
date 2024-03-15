import cvxpy
import cvxpy as cp
import numpy as np
from tqdm.auto import tqdm
import argparse
import time
import matplotlib.pyplot as plt


def repeatable_permutations(n):
    permutations = []

    def generate_permutations(prefix, n):
        if n == 0:
            permutations.append([int(x) for x in prefix])
        else:
            generate_permutations(prefix + '0', n - 1)
            generate_permutations(prefix + '1', n - 1)

    generate_permutations('', n)
    return np.array(permutations)


# ===== exp1 =============


def brute_force_sol(x, y_with_outliers, a, N, args):
    sub_probs = []
    lams = repeatable_permutations(N)
    start_time = time.time()
    for i, lam in tqdm(enumerate(lams), "brute_force_sol: Solving sub-problems...."):

        theta = cp.Variable(1)
        x_train = cp.Parameter(N, value=x)
        y_train = cp.Parameter(N, value=y_with_outliers)
        f_is = []
        constraints = []
        for t, l in enumerate(lam):
            if l:
                f_is.append(cp.square(x_train[t] * theta[0] - y_train[t]))
                constraints.append(cp.square(x_train[t] * theta[0] - y_train[t]) <= a)
            else:
                f_is.append(a)

        objective = cvxpy.Minimize(cvxpy.sum(f_is) + 0.1 * theta[0] ** 2)
        prob = cvxpy.Problem(objective)
        prob.solve()
        if prob.status not in ["infeasible", "unbounded"]:
            sub_probs.append([prob, theta.value, lam])

    objectives = [prob[0].objective.value for prob in sub_probs]
    end_time = time.time()
    min_idx = np.argmin(objectives)
    theta = sub_probs[min_idx][1]
    running_time = end_time - start_time
    return theta[0], running_time


def sccf_sol(x, y_with_outliers, a, N, args):
    import sccf
    start_time = time.time()
    theta = cp.Variable(1)
    x_train = cp.Parameter(N, value=x)
    y_train = cp.Parameter(N, value=y_with_outliers)

    f_is = [sccf.MinExpression(cp.square(x_train[t] * theta[0] - y_train[t]), a)
            for t in range(N)]
    objective = sccf.SumOfMinExpressions(f_is) + 0.1 * theta[0] ** 2
    prob = sccf.Problem(objective)
    prob.solve(step_size=args.step_size, maxiter=args.maxiter, tol=args.tol, verbose=True)
    end_time = time.time()
    running_time = end_time - start_time
    return theta[0].value, running_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=10, type=int, help="# train points")
    parser.add_argument("--n_alias", default=2, type=int, help="# alias points")
    parser.add_argument("--a", default=.2, type=float, help="alpha. See paper-experiment 1")
    parser.add_argument("--theta", default=1, type=float)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--noise", default=0.1, type=float)
    parser.add_argument("--plot_bruteforce", action="store_true", help="Plot BruteForce prediction")
    parser.add_argument("--plot_sccf", action="store_true", help="Plot SCCF prediction")
    parser.add_argument("--step_size", default=0.2, type=float, help="SCCF step size")
    parser.add_argument("--maxiter", default=25, type=int, help="SCCF # iter")
    parser.add_argument("--tol", default=1e-5, type=float, help="SCCF tol")
    args = parser.parse_args()

    np.random.seed(args.seed)
    N = args.N
    true_theta = args.theta
    x = np.random.normal(0, 1, size=(N,))
    z = np.random.normal(0, 1, size=(N,))
    y = true_theta * x  # this is true function we need to estimate of linear regression
    noisy_y = y + args.noise * z  # make it noisy
    indices = np.arange(N)
    np.random.shuffle(indices)
    y_with_outliers = np.copy(y)
    y_with_outliers[indices[:args.n_alias]] *= -1
    a = args.a

    bf_theta, bf_runtime = brute_force_sol(x, y_with_outliers, a, N, args)
    sccf_theta, sccf_runtime = sccf_sol(x, y_with_outliers, a, N, args)

    print('-' * 30)
    print(
        f'Difference with golden theta: brute force: {abs(bf_theta - true_theta)}; sccf: {abs(sccf_theta - true_theta)}')
    print(f'Running time: brute force: {bf_runtime:.2f}s; sccf: {sccf_runtime:.2f}s')
    print('-' * 30)

    # sample x
    sample_x = np.linspace(np.amin(x), np.amax(x), 30)

    if args.plot_bruteforce:
        predicted_y = sample_x * bf_theta
        plt.plot(sample_x, predicted_y, label='bf_theta')

    if args.plot_sccf:
        predicted_y = sample_x * sccf_theta
        plt.plot(sample_x, predicted_y, label='sccf_theta')

    if args.plot_sccf or args.plot_bruteforce:
        plt.scatter(x, y_with_outliers, label='train data')
        plt.legend()
        plt.show()
