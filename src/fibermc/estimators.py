"""Implementations of utilities for sampling fibers and various FMC estimators. 
"""
import functools

import jax 
import jax.numpy as np
import jax.random as npr

import implicit_differentiation
import utils
import geometry_utils

FP64: type = np.float64
pytree: type = dict 

def sample(
    key: np.ndarray,
    bounds: np.ndarray,
    num_fibers: int,
    fiber_length: float,
    dtype: type = np.float32,
) -> np.ndarray:
    """Samples fibers according to a jointly uniform distribution over the starts of
    the fibers and their angle; the endpoints are then determined by the fibers length.

    Parameters
    ----------
    key: np.ndarray
        Psuedo-random number generation key/seed array (via jax.random.PRNGKey).
    bounds: np.ndarray [4,]
        Rectilinear sampling domain specified by a 4-array with elements corresponding to
        (min_x, min_y, max_y, max_y).
    num_fibers: int [>0]
        Strictly positive number of fibers to sample.
    length: float [>0]
        Strictly positive fiber length.
    dtype: type
        Numeric type to use for the fibers (default: FP64).

    Returns
    -------
    fibers: np.ndarray
        An ndarray of shape (num_fibers, 2, 2) containing the fibers
        along axis 0. For each fiber, the start point is the first row 
        of the (2, 2) array and the end point is the second row.
    """
    location_key, angular_key = npr.split(key, 2)
    x_key, y_key = npr.split(location_key, 2)

    starts: np.ndarray = np.array(
        (
            npr.uniform(
                x_key,
                shape=(num_fibers,),
                dtype=dtype,
                minval=bounds[0],
                maxval=bounds[2],
            ),
            npr.uniform(
                y_key,
                shape=(num_fibers,),
                dtype=dtype,
                minval=bounds[1],
                maxval=bounds[3],
            ),
        )
    ).T

    angles: np.ndarray = npr.uniform(
        angular_key,
        shape=(num_fibers,),
        dtype=dtype,
        minval=-np.pi,
        maxval=np.pi,
    )
    ends: np.ndarray = starts + (fiber_length * np.array([np.cos(angles), np.sin(angles)]).T)
    fibers: np.ndarray = np.stack((starts, ends), axis=-2)
    return fibers


def estimate_field_length(
    field: callable, 
    fibers: np.ndarray, 
    params: tuple, 
    negative: bool = True, 
) -> np.ndarray:
    """Estimates the total fiber length for which a given scalar `field` takes on positive/negative
    value.

    Parameters
    ----------
    field: callable[[...], float]
        Scalar real-valued callable which takes auxiliary data `params` and fiber endpoints as
        input argument(s); if fibers are dimension 2 for example, field takes np.ndarrays of size 2
        and `params` to produce a real-valued output.
    fibers: np.ndarray
        np.ndarray of shape (num_fibers, fiber_dim, fiber_dim) of fibers.
    params: pytree
        Auxiliary data provided to the field (e.g., parameters).
    negative: bool
        Estimate the total fiber length for which `field` takes on negative values, if True;
        if instead False, estimate the total fiber length for which `field` takes on positive
        values.

    Returns
    -------
    total_length: np.ndarray
        Nonnegative Monte Carlo estimate of the total fiber length for which `field` takes on negative/positive
        values (negative by default).

    Note: this estimator assumes that `field` changes sign on a lengthscale larger than the length of each
    fiber.
    """
    vector_field: callable = functools.partial(jax.vmap(field, in_axes=(None, 0)), params)
    solver_base: callable = implicit_differentiation.bind_solver(field)
    solver: callable = jax.jit(
        jax.vmap(
            lambda fiber, params: implicit_differentiation.get_interpolant(
                solver_base(np.empty(0), params, fiber), fiber
            ),
            in_axes=(0, None),
        )
    )
    start_points, end_points = fibers[:, 0], fibers[:, 1]
    start_values, end_values = (
        vector_field(start_points).ravel(),
        vector_field(end_points).ravel(),
    )
    start_signs, end_signs = utils.zero_one_sign(start_values), utils.zero_one_sign(end_values)

    if negative:
        negative: float = 0.0
        positive: float = 1.0
    else:
        negative: float = 1.0
        positive: float = 0.0

    count_entire_fiber_cond: np.ndarray = np.logical_and(
        start_signs == negative, end_signs == negative
    )
    count_none_fiber_cond: np.ndarray = np.logical_and(
        start_signs == positive, end_signs == positive
    )
    count_from_start_cond: np.ndarray = np.logical_and(
        start_signs == negative, end_signs == positive
    )
    count_from_end_cond: np.ndarray = np.logical_and(
        start_signs == positive, end_signs == negative
    )

    total_length: np.ndarray = np.zeros(1)

    # count the entire fiber length
    count_all_fibers: np.ndarray = np.where(
        count_entire_fiber_cond.reshape(-1, 1, 1), fibers, np.zeros_like(fibers)
    )
    total_length += jax.vmap(utils.custom_norm)(
        count_all_fibers[:, 1] - count_all_fibers[:, 0]
    ).sum()

    # count from the start of the fiber to the intersection point
    count_from_start: np.ndarray = np.where(
        count_from_start_cond.reshape(-1, 1, 1), fibers, np.zeros_like(fibers)
    )
    total_length += jax.vmap(utils.custom_norm)(
        solver(count_from_start, params) - count_from_start[:, 0]
    ).sum()

    # count from the end of the fiber to the intersection point
    count_from_end: np.ndarray = np.where(
        count_from_end_cond.reshape(-1, 1, 1), fibers, np.zeros_like(fibers)
    )
    total_length += jax.vmap(utils.custom_norm)(
        count_from_end[:, 1] - solver(count_from_end, params)
    ).sum()

    return total_length


def estimate_field_area(
    field: callable, 
    fibers: np.ndarray, 
    params: pytree, 
    negative: bool = True, 
) -> np.ndarray:
    """Estimates the total area for which a scalar `field` takes on positive/negative
    value (negative, by default).

    Parameters
    ----------
    field: callable[[...], float]
        Scalar real-valued callable which takes auxiliary data `params` and fiber endpoints as
        input argument(s); if fibers are dimension 2 for example, field takes np.ndarrays of size 2
        and `params` to produce a real-valued output.
    fibers: np.ndarray
        np.ndarray of shape (num_fibers, fiber_dim, fiber_dim) of fibers.
    params: tuple
        Auxiliary data provided to the field (e.g., parameters).
    negative: bool
        Estimate the total area for which `field` takes on negative values, if True;
        if instead False, estimate the area for which `field` takes on positive
        values.

    Returns
    -------
    total_area: np.ndarray
        Nonnegative Monte Carlo estimate of the total fiber area for which `field` takes on negative/positive
        values (negative by default).

    Note: this estimator assumes that `field` changes sign on a lengthscale larger than the length of each
    fiber.
    """
    cumulative_fiber_length: float = jax.vmap(utils.custom_norm)(
        fibers[:, 1] - fibers[:, 0]
    ).sum()
    total_length: np.ndarray = estimate_field_length(field, fibers, params, negative=negative)
    total_field_area: np.ndarray = total_length / cumulative_fiber_length
    return total_field_area


def estimate_hull_intersection_length(fibers: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Estimates the total fiber length which lies within a provided
    (convex) hull.

    Parameters
    ----------
    fibers: np.ndarray
        np.ndarray of shape (num_fibers, fiber_dim, fiber_dim) containing the
        fibers.
    hull: np.ndarray
        np.ndarray containing line segments along axis 0 which are ordered
        (counter-clockwise by default) to represent a convex hull.

    Example
    -------
    >>> fibers: np.ndarray = np.array([[[-0.25, 0.25], [0.25, 0.25]]])
    >>> hull: np.ndarray = np.array([
                    [0.5, 0.0],
                    [0.5, 0.5],
                    [0.0, 0.5],
                    [0.0, 0.0],
                    [0.5, 0.0],
                    ])
    >>> estimate_hull_intersection_length(fibers, hull)
    >>> 0.25

    Returns
    -------
    estimated_intersection_length: np.ndarray
        Nonnegative estimated length of interesection between `fibers` and `hull`.
    """
    # clip the given fibers so that the resulting 'clipped' fibers all lie within the hull
    intra_hull_fibers: np.ndarray = geometry_utils.clip_inside_convex_hull(fibers, hull)
    estimated_intersection_length: np.ndarray = jax.vmap(utils.custom_norm)(
        intra_hull_fibers[:, 1] - intra_hull_fibers[:, 0]
    ).sum()
    return estimated_intersection_length



def estimate_hull_area(fibers: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Uses fiber Monte Carlo to estimate the area of a convex shape; assuming fibers are sampled from an
    extended domain. See `estimators.estimate_hull_intersection_length`.
    """
    # the cumulative fiber length inside the hull
    intersection_length: np.ndarray = estimate_hull_intersection_length(fibers, hull)

    # the total fiber length (in the hull or not)
    cumulative_fiber_length: np.ndarray = jax.vmap(utils.custom_norm)(
        fibers[:, 1] - fibers[:, 0]
    ).sum()

    # Monte Carlo estimate of the area
    estimated_area: np.ndarray = intersection_length / cumulative_fiber_length
    return estimated_area


@functools.partial(jax.jit, static_argnums=(0, 3))
def clip_to_field(
    field: callable, 
    fibers: np.ndarray, 
    params: pytree, 
    negative: bool = True, 
) -> np.ndarray:
    # vectorize and partially evaluate `field`
    vector_field: callable = functools.partial(jax.vmap(field, in_axes=(None, 0)), params)
    solver: callable = jax.vmap(lambda fiber: implicit_differentiation.bisection_solver(params, fiber, field))
    fiber_dim: int = fibers.shape[-1]

    # the sign of field(x) where x is each fiber endpoint
    start_points, end_points = fibers[:, 0], fibers[:, 1]
    start_values, end_values = (
        vector_field(start_points).ravel(),
        vector_field(end_points).ravel(),
    )
    start_signs, end_signs = utils.zero_one_sign(start_values), utils.zero_one_sign(end_values)

    # (default) 0: field(x) < 0 --- 1: field(x) >= 0
    if negative:
        negative: float = 0.0
        positive: float = 1.0
    else:
        negative: float = 1.0
        positive: float = 0.0

    count_entire_fiber_cond: np.ndarray = np.logical_and(
        start_signs == negative, end_signs == negative
    )
    count_none_fiber_cond: np.ndarray = np.logical_and(
        start_signs == positive, end_signs == positive
    )
    count_from_start_cond: np.ndarray = np.logical_and(
        start_signs == negative, end_signs == positive
    )
    count_from_end_cond: np.ndarray = np.logical_and(
        start_signs == positive, end_signs == negative
    )

    # case: count the entire fiber length
    inside_fibers: np.ndarray = np.where(
        count_entire_fiber_cond.reshape(-1, 1, 1), fibers, np.zeros_like(fibers)
    )

    # case: clip from the start of the fiber to the intersection point
    count_from_start: np.ndarray = np.where(
        count_from_start_cond.reshape(-1, 1, 1), fibers, np.zeros_like(fibers)
    )
    solver_cond: callable = lambda predicates, fibers: jax.vmap(
        lambda predicate, fiber: jax.lax.cond(
            predicate,
            lambda fiber: implicit_differentiation.get_interpolant(solver(fiber[None, :, :]), fiber)[None, :],
            lambda fiber: np.zeros((1, fiber_dim)),
            operand=fiber,
        )
    )(predicates, fibers)
    start_clipped_fibers: np.ndarray = (
        np.dstack(
            (
                count_from_start[:, 0],
                np.squeeze(solver_cond(count_from_start_cond, count_from_start)),
            )
        )
        .swapaxes(2, 1)
        .reshape(-1, 2, fiber_dim)
    )

    # case: count from the end of the fiber to the intersection point
    count_from_end: np.ndarray = np.where(
        count_from_end_cond.reshape(-1, 1, 1), fibers, np.zeros_like(fibers)
    )
    end_clipped_fibers: np.ndarray = (
        np.dstack(
            (
                np.squeeze(solver_cond(count_from_end_cond, count_from_end)),
                count_from_end[:, 1],
            )
        )
        .swapaxes(2, 1)
        .reshape(-1, 2, fiber_dim)
    )

    # aggregate all the valid fibers
    clipped_fibers: np.ndarray = np.vstack(
        (inside_fibers, start_clipped_fibers, end_clipped_fibers)
    )
    return clipped_fibers
