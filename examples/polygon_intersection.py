import dataclasses
from pathlib import Path
from typing import Optional

import jax
import jax.flatten_util
import jax.numpy as np
import jax.random as npr
import matplotlib.pyplot as plt
import shapely
import tyro

import estimators
import geometry_utils


@dataclasses.dataclass
class Config:
    # diagnostics
    report_every: Optional[int] = 10
    plot_every: Optional[int] = 50

    # fmc
    num_fibers: int = 300
    fiber_length: float = 0.5

    # optimization
    num_steps: Optional[int] = 500
    step_size: Optional[float] = 1e-04


def safe_norm(x: np.ndarray, **kwargs) -> np.ndarray:
    return np.linalg.norm(x, **kwargs)


def fiberlen(fibers: np.ndarray) -> np.ndarray:
    return safe_norm(fibers[..., 1] - fibers[..., 0], axis=-1).sum()


def polygon_to_segments(polygon: shapely.Polygon) -> np.ndarray:
    boundary: np.ndarray = np.array(polygon.boundary.xy).T
    segments: np.ndarray = np.stack((boundary[:-1], boundary[1:]), axis=-1).transpose(
        0, 2, 1
    )
    return segments


def clip_to_segments(fibers: np.ndarray, segments: np.ndarray) -> np.ndarray:
    # clip the fibers with respect to the line segments constituting the polygon
    clips, endpoint_sides, has_intersections = geometry_utils.clip_wrt_wall(
        fibers, segments
    )
    clip: np.ndarray = geometry_utils._reduce_clip_params(clips)

    # determine if there are any intersections
    any_intersections: np.ndarray = has_intersections.any(-1)

    is_fully_inside: np.ndarray = (endpoint_sides > 0).all((-1, -2))
    is_fully_outside: np.ndarray = np.logical_and(~is_fully_inside, ~any_intersections)

    # compute the clipping parameters
    clip_parameters: np.ndarray = jax.lax.select(
        is_fully_outside, np.zeros_like(clip), clip
    )

    # apply clipping
    clipped_fiber: np.ndarray = geometry_utils.apply_fiber_clip(fibers, clip_parameters)
    return clipped_fiber


def clip_to_polygon(fibers: np.ndarray, polygon: shapely.Polygon) -> np.ndarray:
    return clip_to_segments(fibers, polygon_to_segments(polygon))


def plot_fibers(fibers: np.ndarray, ax, **kwargs) -> None:
    for fiber in fibers:
        if kwargs.get("endpoints", True):
            ax.scatter(fiber[:, 0], fiber[:, 1], c="tab:red", s=20)
        ax.plot(
            fiber[:, 0],
            fiber[:, 1],
            c=kwargs.get("body_color", "tab:blue"),
            alpha=kwargs.get("alpha", 0.35),
        )


def show(polygons: tuple[shapely.Polygon], fibers: np.ndarray, save_path: Path) -> None:
    polygon, polygon_b = polygons

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(*polygon.exterior.xy, c="tab:blue")
    axs[0].plot(*polygon_b.exterior.xy, c="tab:green")
    axs[0].fill_between(*polygon.exterior.xy, color="tab:blue", alpha=0.2)
    axs[0].fill_between(*polygon_b.exterior.xy, color="tab:green", alpha=0.2)
    axs[1].plot(*polygon.exterior.xy, c="tab:blue")
    axs[1].fill_between(*polygon.exterior.xy, color="tab:blue", alpha=0.2)
    axs[1].plot(*polygon_b.exterior.xy, c="tab:green")
    axs[1].fill_between(*polygon_b.exterior.xy, color="tab:green", alpha=0.2)
    lim_buffer: float = 0.3
    axs[0].set_xlim(
        np.array(
            [
                np.array(polygon.exterior.xy[0]).min(),
                np.array(polygon_b.exterior.xy[0]).min(),
            ]
        ).min()
        - lim_buffer,
        np.array(
            [
                np.array(polygon.exterior.xy[0]).max(),
                np.array(polygon_b.exterior.xy[0]).max(),
            ]
        ).max()
        + lim_buffer,
    )
    axs[0].set_ylim(
        np.array(
            [
                np.array(polygon.exterior.xy[1]).min(),
                np.array(polygon_b.exterior.xy[1]).min(),
            ]
        ).min()
        - lim_buffer,
        np.array(
            [
                np.array(polygon.exterior.xy[1]).max(),
                np.array(polygon_b.exterior.xy[1]).max(),
            ]
        ).max()
        + lim_buffer,
    )
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xlim(
        np.array(
            [
                np.array(polygon.exterior.xy[0]).min(),
                np.array(polygon_b.exterior.xy[0]).min(),
            ]
        ).min()
        - lim_buffer,
        np.array(
            [
                np.array(polygon.exterior.xy[0]).max(),
                np.array(polygon_b.exterior.xy[0]).max(),
            ]
        ).max()
        + lim_buffer,
    )
    axs[1].set_ylim(
        np.array(
            [
                np.array(polygon.exterior.xy[1]).min(),
                np.array(polygon_b.exterior.xy[1]).min(),
            ]
        ).min()
        - lim_buffer,
        np.array(
            [
                np.array(polygon.exterior.xy[1]).max(),
                np.array(polygon_b.exterior.xy[1]).max(),
            ]
        ).max()
        + lim_buffer,
    )
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plot_fibers(fibers, axs[0])
    clipped: np.ndarray = jax.vmap(clip_to_polygon, in_axes=(0, None))(fibers, polygon)
    clipped: np.ndarray = jax.vmap(clip_to_polygon, in_axes=(0, None))(
        clipped, polygon_b
    )
    clipped = clipped[np.all(clipped[:, 1] != clipped[:, 0], axis=-1), ...]
    plot_fibers(clipped, axs[1], body_color="tab:red", endpoints=True, alpha=1.0)
    plt.savefig(save_path)
    plt.close()


def segments_to_polygon(segments: np.ndarray) -> shapely.Polygon:
    vertices: np.ndarray = segments[:, 0, :]
    return shapely.Polygon(vertices)


def rotation(theta: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


def transform_shape(params: tuple[np.ndarray], segments: np.ndarray) -> np.ndarray:
    theta, translation = params
    mean: np.ndarray = segments.mean(0)
    R: np.ndarray = rotation(theta)
    out: np.ndarray = R[np.newaxis, ...] @ (segments - mean) + translation + mean
    return out


def show_configuration(
    segments: tuple[np.ndarray],
    params: tuple[np.ndarray],
    fibers: np.ndarray,
    save_path: Path,
) -> None:
    polygon, polygonb = (
        segments_to_polygon(segments[0]),
        segments_to_polygon(transform_shape(params, segments[1])),
    )

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.plot(*polygon.exterior.xy, c="tab:blue")
    axs.fill_between(*polygon.exterior.xy, color="tab:blue", alpha=0.2)
    axs.plot(*polygonb.exterior.xy, c="tab:green")
    axs.fill_between(*polygonb.exterior.xy, color="tab:green", alpha=0.2)

    lim_buffer: float = 0.3
    axs.set_xlim(
        np.array(
            [
                np.array(polygon.exterior.xy[0]).min(),
                np.array(polygonb.exterior.xy[0]).min(),
            ]
        ).min()
        - lim_buffer,
        np.array(
            [
                np.array(polygon.exterior.xy[0]).max(),
                np.array(polygonb.exterior.xy[0]).max(),
            ]
        ).max()
        + lim_buffer,
    )
    axs.set_ylim(
        np.array(
            [
                np.array(polygon.exterior.xy[1]).min(),
                np.array(polygonb.exterior.xy[1]).min(),
            ]
        ).min()
        - lim_buffer,
        np.array(
            [
                np.array(polygon.exterior.xy[1]).max(),
                np.array(polygonb.exterior.xy[1]).max(),
            ]
        ).max()
        + lim_buffer,
    )
    vector_clip: callable = jax.vmap(clip_to_segments, in_axes=(0, None))

    clipped: np.ndarray = vector_clip(
        vector_clip(fibers, segments[0]), transform_shape(params, segments[1])
    )
    clipped = clipped[np.all(clipped[:, 1] != clipped[:, 0], axis=-1), ...]
    plot_fibers(clipped, axs, body_color="tab:red", endpoints=True, alpha=1.0)
    plt.savefig(save_path)
    plt.close()


def main(config: Config):
    vertices: np.ndarray = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.3],
            [1.0, 0.0],
            [0.7, 0.5],
            [1.0, 1.0],
            [0.5, 0.7],
            [0.0, 1.0],
            [0.3, 0.5],
        ]
    )
    vertices_b: np.ndarray = np.array(
        [
            [0.5, 1.0],
            [1.0, 1.0],
            [1.0, 1.5],
            [0.5, 1.5],
        ]
    )
    polygon = shapely.Polygon(vertices)
    polygon_b = shapely.Polygon(vertices_b)

    # sample fibers
    key: np.ndarray = npr.PRNGKey(0)
    boundary_buffer: float = 0.0
    bounds: np.ndarray = np.array(
        [
            0.0 - boundary_buffer,
            0.0 - boundary_buffer,
            1.0 + boundary_buffer,
            1.0 + boundary_buffer,
        ]
    )
    fibers: np.ndarray = estimators.sample(
        key, bounds, config.num_fibers, config.fiber_length
    )

    polygons: tuple[shapely.Polygon] = (polygon, polygon_b)
    save_path: Path = Path("polygon_intersection.png")
    # show(polygons, fibers, save_path)
    # print(f"Saved demo figure to: {str(save_path)}")

    polygon_segments: np.ndarray = polygon_to_segments(polygon)
    polygonb_segments: np.ndarray = polygon_to_segments(polygon_b)

    params: tuple[np.ndarray] = (np.array(0.0), np.zeros(2))

    # show_configuration(
    #     (polygon_segments, polygonb_segments),
    #     params,
    #     fibers,
    #     Path(f"intersection_init.png"),
    # )

    def collapsed(fiber: np.ndarray) -> np.ndarray:
        return (fiber[:, 0, :] == fiber[:, 1, :]).all(-1)

    def objective(params: tuple[np.ndarray]) -> np.ndarray:
        transformed: np.ndarray = transform_shape(params, polygonb_segments)
        vector_clip: callable = jax.vmap(clip_to_segments, in_axes=(0, None))

        in_p1: np.ndarray = vector_clip(fibers, polygon_segments)
        in_p2: np.ndarray = vector_clip(fibers, transformed)

        in_both: np.ndarray = vector_clip(in_p1, transformed)
        intersection_estimate: np.ndarray = fiberlen(in_both)
        # union_estimate: np.ndarray = fiberlen(in_p2)
        return -(intersection_estimate).squeeze()

    # core optimization loop
    gradient_fn: callable = jax.jit(jax.value_and_grad(objective))
    gradient_fn_nojit: callable = jax.value_and_grad(objective)

    current_objective, gradient = gradient_fn(params)
    current_objective, gradient_nojit = gradient_fn_nojit(params)
    print(f"gradient: {gradient}")
    print(f"gradient (no jit): {gradient_nojit}")

    import sys

    sys.exit(0)

    for i in range(config.num_steps):
        current_objective, gradient = gradient_fn(params)
        params = jax.tree.map(lambda p, g: p - config.step_size * g, params, gradient)

        gradient_flat: np.ndarray = jax.flatten_util.ravel_pytree(gradient)[0]
        gradient_norm: np.ndarray = np.linalg.norm(gradient_flat)

        if i % config.report_every == 0:
            print(
                f"Iteration [{i:04d}/{config.num_steps:04d}]\tIntersection: {-current_objective.item():.3f}\tGrad Norm: {gradient_norm.item():.1f}"
            )

        if i % config.plot_every == 0:
            show_configuration(
                (polygon_segments, polygonb_segments),
                params,
                fibers,
                Path(f"intersection_{i}.png"),
            )


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
