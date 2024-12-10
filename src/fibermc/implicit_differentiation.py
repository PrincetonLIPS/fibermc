"""This module contains procedures for implicit differentiation setups in
conjunction with fiber sampling applications.
"""

import functools
from typing import TypeVar

import jax
import jax.numpy as np
from jaxopt.implicit_diff import custom_root

from fibermc.jax_utils import divide00

pytree: type = TypeVar("Pytree")


# @functools.partial(jax.jit, static_argnums=(0, 2))
def bisect(f: callable, fiber: np.ndarray, num_iterations: int = 10) -> float:
    interpolant: callable = lambda x: fiber[0] + x * (fiber[1] - fiber[0])
    h: callable = lambda x: f(interpolant(x))

    # standardize so the 'left' endpoint has negative value
    endpoints: np.ndarray = jax.lax.cond(
        h(0.0) > 0.0,
        lambda _: np.array([1.0, 0]),
        lambda _: np.array([0.0, 1.0]),
        operand=None,
    )

    def _bisect(endpoints: np.ndarray) -> float:
        left, right = endpoints
        midpoint: float = (left + right) / 2.0
        return jax.lax.cond(
            h(midpoint) < 0.0,
            lambda _: np.array([midpoint, right]),
            lambda _: np.array([left, midpoint]),
            operand=midpoint,
        )

    for _ in range(num_iterations):
        endpoints: np.ndarray = _bisect(endpoints)

    return endpoints[0]


def get_interpolant(alpha: np.ndarray, fiber: np.ndarray) -> np.ndarray:
    return fiber[0] + alpha * (fiber[1] - fiber[0])


def bind_optimality_condition(f: callable) -> np.ndarray:
    def optimality_condition(
        x: np.ndarray, params: dict, fiber: np.ndarray
    ) -> np.ndarray:
        """Computes the the scalar-value of the `constraint function` which takes on the
        value zero when the constrain is satisfied.

        Parameters
        ----------
        x: np.ndarray
            np.ndarray (of length 1), representing the interpolant point in the domain
            of `scalar_field` to be evaluated. This is a point defined such that:
            fiber[0] + x * (fiber[1] - fiber[0]) lies on the zero set of the scalar
            field.
        params: tuple
            auxiliary parameters for `scalar_field`.
        fiber: np.ndarray
            fiber included as the second constraint; the constraint is zero when
            `x` lies on this fiber, otherwise the value of the constraint is the
            distance between `x` and the fiber. TODO this isn't actually used

        Returns
        -------
        constraint_value: np.ndarray
            the value of the scalar field evaluated at get_interpolant(x, fiber)
        """
        z: np.ndarray = get_interpolant(x, fiber)
        constraint_value: np.ndarray = f(params, z)
        return constraint_value

    return optimality_condition


def bind_solver(f: callable) -> callable:
    @custom_root(bind_optimality_condition(f))
    def bisection_solver(
        x_init: np.ndarray, params: dict, fiber: np.ndarray
    ) -> np.ndarray:
        """Computes the (intersection) point that is (1) on the line segment represented
        by `fiber` and (2) at which the scalar field takes on the value zero.

        Parameters
        ----------
        x_init: np.ndarray
            unused but necessary for signature of jaxopt.custom_root.
        params: tuple
            auxiliary parameters for the bound callable `f`.
        fiber: np.ndarray
            line segment for which the scalar_field, when evaluated at each endpoint,
            returns a value with opposing sign.

        Returns
        -------
        fixed_point: np.ndarray
            intersection point which (1) lies on the line segment specified by `fiber`
            and (2) satisfies f(params, fixed_point) == 0.
        """
        x: np.ndarray = bisect(functools.partial(f, params), fiber)
        return x

    return bisection_solver


def bisection_constraint(
    f: callable, x: np.ndarray, params: tuple, fiber: np.ndarray
) -> np.ndarray:
    """Computes the np.ndarray of length 2 representing the vector-value of the
    `constraint function` which takes on the value of the zero-vector when
    the constrain is satisfied.

    Parameters
    ----------
    scalar_field: callable
        scalar field function whose level-set at zero is included as one of
        the constraints; the field is evaluated at `x` and `params`.
    x: np.ndarray
        np.ndarray (of length 2, nominally), representing the point in the domain
        of `scalar_field` to be evaluated.
    params: tuple
        auxiliary parameters for `scalar_field`.
    fiber: np.ndarray
        fiber included as the second constraint; the constraint is zero when
        `x` lies on this fiber, otherwise the value of the constraint is the
        distance between `x` and the fiber.
    Returns
    -------
    constraint_value: np.ndarray
        first element is the value of the first constraint (i.e., the value of
        the scalar field evaluated at x); second element is the distance between
        `x` and the fiber, it is zero when `x` is coincident with the fiber.
    Note: where the constraint is satisfied, this means `x` is both coincident
    with `fiber` and lies on the (zero) level-set of the function `scalar_field`.
    """
    z: np.ndarray = get_interpolant(x, fiber)
    field_constraint: float = np.squeeze(f(params, z))
    return field_constraint


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def bisection_solver(params: pytree, fiber: np.ndarray, f: callable) -> np.ndarray:
    """Computes the (intersection) point that is (1) on the line segment represented
    by `fiber` and (2) at which the scalar field takes on the value zero.

    Parameters
    ----------
    f: callable
        function which should return a real-valued scalar when applied like:
        f(params, x).
    fiber: np.ndarray
        line segment for which the scalar_field `f`, when evaluated at each endpoint,
        returns a value with opposing sign.
    params: tuple
        auxiliary parameters for `f`.

    Returns
    -------
    fixed_point: np.ndarray
        intersection point which (1) lies on the line segment specified by `fiber`
        and (2) satisfies f(params, fixed_point) == 0.
    See also: src.implicit_differentiation.bisection_solver{forward, backward}
    """
    fixed_point: np.ndarray = bisect(functools.partial(f, params), fiber)
    return fixed_point


def bisection_solver_forward(
    params: tuple, fiber: np.ndarray, f: callable
) -> np.ndarray:
    """Uses `src.implicit_differentiation.bisection_solver` to determine the
    intersection between the fiber and the zero level-set of the scalar field.

    Parameters
    ----------
    scalar_field: callable
        function which should return a real-valued scalar when applied like:
        scalar_field(params, x).
    fiber: np.ndarray
        line segment for which the scalar_field, when evaluated at each endpoint,
        returns a value with opposing sign.
    params: tuple
        auxiliary parameters for `scalar_field`.
    Returns
    -------
    payload: tuple
        first element is the np.ndarray containing the intersection point; second element
        is the collection of residuals used in the backward pass to compute the vjp.
    """
    # determine the intersection point
    fixed_point: np.ndarray = bisection_solver(params, fiber, f)

    # collect the residuals to be used in the backward pass
    residuals: tuple = (params, fixed_point, fiber)

    payload: tuple = (fixed_point, residuals)
    return payload


def bisection_solver_backward(
    f: callable, residuals: tuple, incoming_gradient: np.ndarray
) -> np.ndarray:
    """Computes the vector-Jacobian project associated with the bisection solver
    procedure, using implicit differentiation.

    Parameters
    ----------
    scalar_field: callable
        function which should return a real-valued scalar when applied like:
        scalar_field(params, x).
    fiber: np.ndarray
        line segment for which the scalar_field, when evaluated at each endpoint,
        returns a value with opposing sign.
    residuals: tuple
        first element is the parameters associated with the scalar_field (assumed
        fixed) when the bisection solver was invoked; second element is the intersection
        point.
    incoming_gradient: np.ndarray
        gradient signal arising from some downstream (from the perspective of the
        forward pass, that is) computation for which autodiff has already produced
        derivative values; these are used to correctly proceed with the chain rule
        back upstream of wherever `bisection_solver` was called.
    Returns
    -------
    final_vjp: np.ndarray
        array containing the local derivatives associated with the bisection solver
        and multiplied with the incoming (dowstream) derivatives.
    """
    # unpack residuals
    params, fixed_point, fiber = residuals

    # f's univariate analogues
    f_params: callable = lambda _params: bisection_constraint(
        f, fixed_point, _params, fiber
    )
    f_spatial: callable = lambda _x: bisection_constraint(f, _x, params, fiber)

    # partial vjps (w.r.t. params and the spatial variable)
    _, vjp_params = jax.vjp(f_params, params)
    _, vjp_spatial = jax.vjp(f_spatial, fixed_point)

    # solve for the intermediate vjp
    jacobian_f_fn: callable = jax.jacobian(f_spatial)
    jacobian_f: np.ndarray = jacobian_f_fn(fixed_point)

    A: np.ndarray = jacobian_f.T
    b: np.ndarray = incoming_gradient
    intermediate_vjp: np.ndarray = -1.0 * divide00(b, A)
    final_vjp: np.ndarray = vjp_params(intermediate_vjp)

    return (final_vjp[0], None)


# --- solver custom vjp binding
bisection_solver.defvjp(bisection_solver_forward, bisection_solver_backward)
