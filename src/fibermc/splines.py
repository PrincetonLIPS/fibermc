"""B-Spline Utility functions.

Ported directly from: https://github.com/PrincetonLIPS/Varmint_dev/blob/main/varmint/geometry/bsplines.py.
"""

import jax.numpy as np

FLOAT_T: type = np.float32


def divide00(num, denom):
    force_zero = np.logical_and(num == 0, denom == 0)
    return np.where(force_zero, FLOAT_T(0.0), num) / np.where(
        force_zero, FLOAT_T(1.0), denom
    )


def default_knots(degree: int, num_control: int):
    return np.hstack(
        [np.zeros(degree), np.linspace(0, 1, num_control - degree + 1), np.ones(degree)]
    )


def construct_basis(u: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """
    u: [batch_dim, 1]
    knots: [K,]
    degree: []

    Allocations:
        - [batch_dim, 1] * 2
        - [1, O(K)] * 5
        - [batch_dim, 1, O(k)] * 5
    """
    u1d = np.atleast_1d(u)  # [batch_dim, 1]
    u = np.expand_dims(u1d, -1)  # [batch_dim, 1, 1]
    k = np.expand_dims(knots, tuple(range(u1d.ndim)))  # [1, K]
    k = np.where(
        k == knots[-1], knots[-1] + np.finfo(FLOAT_T).eps, k
    )  # [1, K] (could allocate [3, K])
    basis: np.ndarray = (k[..., :-1] <= u) * (
        u < k[..., 1:]
    ) + 0.0  # [1, (K-1)] (could allocate [3, (K-1)])

    for d in range(1, degree + 1):
        v0_num = basis[..., :-1] * (u - k[..., : -d - 1])  # [batch_dim, 1, K - degree]
        v0_denom = k[..., d:-1] - k[..., : -d - 1]  # [1, K - degree]
        v0 = divide00(v0_num, v0_denom)  # [batch_dim, 1, K - degree]
        v1_num = basis[..., 1:] * (k[..., d + 1 :] - u)  # [batch_dim, 1, K - degree]
        v1_denom = k[..., d + 1 :] - k[..., 1:-d]  # [1, K - degree]
        v1 = divide00(v1_num, v1_denom)  # [batch_dim, 1, K - degree]
        basis = v0 + v1  # [batch_dim, 1, K - degree]

    return basis


def construct_basis2d(
    u: np.ndarray, knots: tuple[np.ndarray], degree: int
) -> np.ndarray:
    """
    u: [batch_dim, 2]
    knots: ([num_control + degree + 1, num_control + degree + 1])
    degree: []
    """
    u = np.atleast_2d(u)  # [batch_dim, 2]
    x_knots, y_knots = knots  # [Kx, Ky]
    basis_x: np.ndarray = construct_basis(u[:, 0], x_knots, degree)[
        :, :, np.newaxis
    ]  # [batch_dim, 1, Kx - (degree + 1), 1]
    basis_y: np.ndarray = construct_basis(u[:, 1], y_knots, degree)[
        :, np.newaxis, :
    ]  # [batch_dim, 1, 1, Ky - (degree + 1)]
    basis: np.ndarray = (
        basis_x * basis_y
    )  # [batch_dim, 1, Kx - (degree + 1), Ky - (degree + 1)]
    return basis


def bspline(
    u: np.ndarray, control: np.ndarray, knots: np.ndarray, degree: int
) -> np.ndarray:
    return construct_basis(u, knots, degree) @ control


def bspline2d(
    u: np.ndarray, control: np.ndarray, knots: tuple[np.ndarray], degree: int
) -> np.ndarray:
    """
    control [c_0, c_1, ...]
    u [batch_dim, 2]
    knots [c_0, c_1, ...] + degree + 1
    degree []
    """
    basis: np.ndarray = construct_basis2d(u, knots, degree)  # [batch_dim, 1, c_0, c_1]
    out: np.ndarray = np.tensordot(basis, control, ((1, 2), (0, 1)))  # [batch_dim,]
    return out
