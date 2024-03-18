"""
=============== Author Implementation SCCF ======================
original code: https://github.com/cvxgrp/sccf/blob/master/sccf/sccf.py

=============== Integrated with Brute Force Solver [our implementation] ==============
Important: you may need mosek license. See instruction at https://www.mosek.com/resources/getting-started/
"""
import numbers
import warnings

import cvxpy
import numpy as np
import cvxpy as cp
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.constants import Constant

from scipy.optimize import fmin_l_bfgs_b, check_grad
from scipy.sparse import issparse


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


def asscalar(x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return x.item()


def flatten_vars(vars):
    x = np.empty(0)
    for v in vars:
        x = np.append(x, v.value.flatten(order="F"))
    return x


def unflatten_vars(x, vars):
    i = 0
    for v in vars:
        num = int(np.prod(v.shape))
        v.value = x[i:i + num].reshape(v.shape, order="F")
        i += num


def flatten_grads(grads, vars):
    grad = np.empty(0)
    for v in vars:
        try:
            grad_v = grads[v]
        except KeyError:
            grad_v = np.zeros(v.shape)
        if issparse(grad_v):
            grad_v = grad_v.todense()
        grad = np.append(grad, grad_v.flatten(order="F"))
    return grad


class MinExpression:
    """
    A class used to represent the minimum of a cvxpy expression and a number.

    Attributes
    ----------
    expr : cvxpy Expression
    a : numbers.Number
    value : double
        The evaluation of the expression
    """

    def __init__(self, expr, a):
        """Forms the MinExpression min(expr, a).

        Parameters
        ----------
        expr : cvxpy Expression
        a : numbers.Number
        """
        if isinstance(expr, numbers.Number):
            expr = Constant(expr)
        assert isinstance(expr, Expression), "expr must be a cvxpy Expression"
        assert isinstance(a, numbers.Number), "a must be a number"
        assert expr.is_convex(), "expr must be convex"
        self.expr = expr
        self.a = a

    @property
    def value(self):
        """Evaluates min(expr, a)."""
        return asscalar(min(self.expr.value, self.a))


class SumOfMinExpressions:
    """
    A class used to represent a sum of MinExpressions.

    Attributes
    ----------
    min_exprs : list
        List of MinExpression's.
    value : double
        The numerical evaluation of the object.

    Methods
    -------
    __add__(e)
        Adds the object to an Expression, MinExpression, SumOfMinExpressions, or numbers.Number
    """

    def __init__(self, min_exprs):
        """
        Parameters
        ----------
        min_exprs : list
            The list of MinExpression's.
        """
        self.min_exprs = min_exprs

    @property
    def num_exprs(self):
        return len(self.min_exprs)

    def __add__(self, e):
        """Adds the SumOfMinExpressions to another object.

        Parameters
        ----------
        e : Expression, MinExpression, SumOfMinExpressions, or numbers.Number

        Raises
        ------
        ValueError
            If e is not a supported type.

        Returns
        -------
        expression : SumOfMinExpressions
            The result of the addition.
        """
        if isinstance(e, MinExpression):
            self.min_exprs += [e]
            return self
        elif isinstance(e, SumOfMinExpressions):
            self.min_exprs += e.min_exprs
            return self
        elif isinstance(e, numbers.Number):
            self.min_exprs += minimum(e, e).min_exprs
            return self
        elif isinstance(e, Expression):
            self.min_exprs += minimum(e, float("inf")).min_exprs
            return self
        else:
            raise ValueError("type %s not supported in __add__" % type(e))

    __radd__ = __add__

    @property
    def value(self):
        """The value of the expression."""
        return asscalar(np.sum([min_expr.value for min_expr in self.min_exprs]))


def minimum(expr, a):
    """Forms a SumOfMinExpressions object that represents min(expr, a).

    Parameters
    ----------
    expr : cvxpy Expression
    a : numbers.Number
    """
    return SumOfMinExpressions([MinExpression(expr, a)])


class Problem:
    """
    A class used to represent a problem of
    minimizing a sum of clipped convex functions.

    Attributes
    ----------
    objective : SumOfMinExpressions
        The objective, which must be a SumOfMinExpressions.
    constraints : list
        A list of cvxpy constraints.
    vars_ : list
        The cvxpy Variables involved in the problem.
    """

    def __init__(self, objective, constraints=[]):
        """
        Parameters
        ----------
        objective : SumOfMinExpressions
        constraints : list
        """
        assert isinstance(objective, SumOfMinExpressions), "objective must be a SumOfMinExpressions"
        self.objective = objective
        self.constraints = constraints
        self.vars_ = []
        for min_expr in self.objective.min_exprs:
            self.vars_ += min_expr.expr.variables()
        for constr in self.constraints:
            self.vars_ += constr.variables()
        self.vars_ = list(set(self.vars_))

    def solve(self, method="alternating", *args, **kwargs):
        """Approximately solve the problem using convex-concave or L-BFGS.

        Parameters
        ==========
        method : str
            cvx_ccv, alternating, or lbfgs
        args, kwargs
        """
        if method == "alternating":
            return self._solve_alternating(*args, **kwargs)
        elif method == "cvx_ccv":
            return self._solve_cvx_ccv(*args, **kwargs)
        elif method == "lbfgs":
            if len(self.constraints) > 0:
                raise ValueError("Cannot use L-BFGS when there are constraints. Please use cvx_ccv or alternating")
            else:
                return self._solve_lbfgs(*args, **kwargs)
        elif method == "brute_force":
            return self._solve_brute_force(*args, **kwargs)
        raise NotImplementedError(f"method {method} not supported")

    def _solve_brute_force(self, verbose=False, *args, **kwargs):
        sub_probs = []
        expressions = {
            'clip_exprs': [],
            'clip_as': [],
            'others': []
        }
        for exp in self.objective.min_exprs:
            exp: MinExpression
            if exp.a != float("inf") and exp.a is not exp.expr:
                expressions['clip_exprs'].append(exp.expr)
                expressions['clip_as'].append(exp.a)
            else:
                expressions['others'].append(exp.expr)
        N = len(expressions['clip_exprs'])
        self.lam = cvxpy.Parameter(N)
        if N > 14:
            warnings.warn(f"Number of clipped expression is too large. You may expect the complexity of 2^{N}")
        lams = repeatable_permutations(N)
        min_obj_vals = float('inf')
        optimal_vals = float('inf')
        if verbose:
            from tqdm.auto import tqdm
            iteration = tqdm(enumerate(lams), "brute_force_sol: Solving sub-problems....")
        else:
            iteration = enumerate(lams)
        for i, lam in iteration:
            relaxed_fs = []
            constraints = [*self.constraints]
            for t, (l, un_clipped, a) in enumerate(zip(lam, expressions['clip_exprs'], expressions['clip_as'])):
                if l:
                    # add expression with constraint
                    relaxed_fs.append(un_clipped)
                    constraints.append(un_clipped <= a)
                else:
                    # add alpha
                    relaxed_fs.append(a)
            objective = cvxpy.sum(relaxed_fs) + cvxpy.sum(expressions['others'])
            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve()
            if prob.status not in ["infeasible", "unbounded"]:
                if min_obj_vals > objective.value:
                    ref_vars = prob.variables()
                    optimal_vals = [var.value for var in ref_vars]
                    optimal_vals = (ref_vars, optimal_vals, lam)
                    min_obj_vals = objective.value
        if min_obj_vals != float('inf'):
            ## assign the optimal_vals to the variable value
            ref_vars, optimal_vals, lam = optimal_vals
            for ref_var, optimal_val in zip(ref_vars, optimal_vals):
                ref_var: cvxpy.Variable
                ref_var.project_and_assign(optimal_val)
                self.lam.project_and_assign(lam)

        return min_obj_vals

    def _solve_alternating(self, step_size=0.2, maxiter=25, tol=1e-5, verbose=False,
                           warm_start=False, warm_start_lam=None, **kwargs):
        """Approximately solves the Problem using an alternating minimization procedure.

        Parameters
        ----------
        step_size : double
            Step size
        maxiter : int
            Maximum number of iterations (default = 10)
        tol : double
            Numerical tolerance for stopping condition (default = 1e-5)
        verbose : bool
            Whether or not to print information (default = False)
        warm_start : bool
            Whether or not to warm start x (default = False)
        warm_start_lam : np.array
            Warm start value for lam (default = None)
        **kwargs
            Keyword arguments to be sent to cvxpy solve function

        Returns
        -------
        info : dict
            Dictionary of solver information
        """
        m = self.objective.num_exprs
        lam = 0.5 * np.ones(m)
        if warm_start_lam is not None:
            lam = warm_start_lam

        alphas = np.array([min_expr.a for min_expr in self.objective.min_exprs])
        Ls, lams = [], []

        lam_cp = cp.Parameter(m, nonneg=True)
        objective = 0.0
        for i, min_expr in enumerate(self.objective.min_exprs):
            if min_expr.a == np.inf:
                objective += min_expr.expr
            else:
                objective += lam_cp[i] * min_expr.expr
                objective += (1 - lam_cp[i]) * min_expr.a
        prob = cp.Problem(cp.Minimize(objective), self.constraints)

        lam = np.clip(lam, 0.0, 1.0) # there is a possible bug of negative lam
        for k in range(maxiter):
            # x step, skip if warm_start=True and k=0
            if not warm_start or k > 0:
                lam_cp.value = lam
                result = prob.solve(**kwargs)
                if prob.status == 'unbounded' or prob.status == 'infeasible':
                    raise ValueError("Un-clipped problem is %s." % prob.status)

            # lam step
            lams.append(lam)
            fk = np.array([min_expr.expr.value for min_expr in self.objective.min_exprs])
            lam_next = lam - step_size * np.sign(fk - alphas)
            lam_next[alphas == np.inf] = 1.0
            lam_next = np.clip(lam_next, 0.0, 1.0)

            stopping_condition = np.linalg.norm(lam_next - lam, np.inf)

            lam = lam_next.astype(np.double)
            L = self.objective.value
            Ls.append(L)

            if verbose:
                print("%04d | %4.4e | %4.4e" % (k + 1, L, stopping_condition))

            if stopping_condition < tol:
                if verbose:
                    print("Terminated (stopping condition satisfied).")
                break

        if verbose and k == maxiter - 1:
            print("Terminated (maximum number of iterations reached).")

        info = {"final_objective_value": L, "objective_values": Ls, "stopping_condition": stopping_condition,
                "iters": k + 1, "lams": lams}

        return info

    def _solve_cvx_ccv(self, maxiter=10, tol=1e-5, verbose=False, warm_start=False, **kwargs):
        """Approximately solves the Problem using the convex-concave procedure.

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations (default = 10)
        tol : double
            Numerical tolerance for stopping condition (default = 1e-5)
        verbose : bool
            Whether or not to print solver information (default = False)
        warm_start : bool
            Whether or not to warm start (default = False, which warm starts with un-clipped problem)
        **kwargs
            Keyword arguments to be sent to cvxpy solve function

        Returns
        -------
        result : objective value
        """
        if not warm_start:
            objective = cp.sum([min_expr.expr for min_expr in self.objective.min_exprs])
            prob = cp.Problem(cp.Minimize(objective), self.constraints)
            prob.solve(**kwargs)

        prev_L = float("inf")
        for k in range(maxiter):
            objective = 0.0
            for min_expr in self.objective.min_exprs:
                g = max(min_expr.expr.value - min_expr.a, 0.0)
                objective += min_expr.expr
                objective -= g
                if g > 0.0:
                    expr_grad = min_expr.expr.grad
                    for var in self.vars_:
                        if isinstance(expr_grad[var], numbers.Number):
                            grad = np.array(expr_grad[var])
                        else:
                            grad = np.array(expr_grad[var].todense())
                        var_minus_prev = var - var.value
                        grad = grad.reshape(var_minus_prev.shape)
                        objective -= cp.sum(cp.multiply(grad, var_minus_prev))
            prob = cp.Problem(cp.Minimize(objective), self.constraints)
            result = prob.solve(**kwargs)
            L = self.objective.value
            if prev_L - L <= tol:
                break
            prev_L = L
            if verbose:
                print("%04d | %4.8e" % (k + 1, prev_L))
        return L

    def _solve_lbfgs(self, warm_start=False, **kwargs):
        """Use L-BFGS."""

        if not warm_start:
            objective = cp.sum([min_expr.expr for min_expr in self.objective.min_exprs])
            prob = cp.Problem(cp.Minimize(objective), self.constraints)
            prob.solve()

        m = len(self.objective.min_exprs)
        n = sum([int(np.prod(v.shape)) for v in self.vars_])

        def func(x):
            lam = x[:m]
            x = x[m:]
            unflatten_vars(x, self.vars_)
            grad_lam = np.zeros(m)
            grad_x = np.zeros(n)
            L = 0.0
            for i in range(m):
                min_expr = self.objective.min_exprs[i]
                fi = min_expr.expr.value
                ai = min_expr.a
                g = flatten_grads(min_expr.expr.grad, self.vars_)
                if ai == np.inf:
                    grad_x += g
                    grad_lam[i] = 0.0
                    L += fi
                else:
                    grad_x += lam[i] * g
                    grad_lam[i] = fi - ai
                    L += lam[i] * fi + (1 - lam[i]) * ai
            grad = np.append(grad_lam, grad_x)

            return L, grad

        x0 = np.append(.5 * np.ones(m), flatten_vars(self.vars_))
        bounds = [(0.0, 1.0) for _ in range(m)]
        bounds += [(-np.inf, np.inf) for _ in range(n)]

        check = check_grad(lambda x: func(x)[0], lambda x: func(x)[1], x0)
        assert check <= 1e-4, "check_grad did not work, %.3f > 1e-4" % check

        result = fmin_l_bfgs_b(func, x0, bounds=bounds, **kwargs)

        unflatten_vars(result[0][m:], self.vars_)

        return self.objective.value
