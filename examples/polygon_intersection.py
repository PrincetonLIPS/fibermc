from pathlib import Path

import jax 
import jax.numpy as np
import jax.random as npr
import matplotlib.pyplot as plt 
import shapely 

import estimators
import geometry_utils

def polygon_to_segments(polygon: shapely.Polygon) -> np.ndarray: 
    boundary: np.ndarray = np.array(polygon.boundary.xy).T
    segments: np.ndarray = np.stack((boundary[:-1], boundary[1:]), axis=-1).transpose(0, 2, 1)
    return segments

def clip_to_segments(fibers: np.ndarray, segments: np.ndarray) -> np.ndarray: 
    # clip the fibers with respect to the line segments constituting the polygon 
    clips, endpoint_sides, has_intersections = geometry_utils.clip_wrt_wall(fibers, segments)
    clip: np.ndarray = geometry_utils._reduce_clip_params(clips)

    # determine if there are any intersections 
    any_intersections: np.ndarray = has_intersections.any(-1)

    is_fully_inside: np.ndarray = (endpoint_sides > 0).all((-1, -2))
    is_fully_outside: np.ndarray = np.logical_and(~is_fully_inside, ~any_intersections)
    
    # compute the clipping parameters
    clip_parameters: np.ndarray = jax.lax.select(is_fully_outside, np.zeros_like(clip), clip)
    
    # apply clipping 
    clipped_fiber: np.ndarray = geometry_utils.apply_fiber_clip(fibers, clip_parameters)
    return clipped_fiber

def clip_to_polygon(fibers: np.ndarray, polygon: shapely.Polygon) -> np.ndarray: 
    return clip_to_segments(fibers, polygon_to_segments(polygon))

def show(polygons: tuple[shapely.Polygon], fibers: np.ndarray, save_path: Path) -> None: 
    polygon, polygon_b = polygons 
    def plot_fibers(fibers: np.ndarray, ax, **kwargs) -> None: 
        for fiber in fibers: 
            if kwargs.get("endpoints", True):
                ax.scatter(fiber[:, 0], fiber[:, 1], c="tab:red", s=20)
            ax.plot(fiber[:, 0], fiber[:, 1], c=kwargs.get("body_color", "tab:blue"), alpha=kwargs.get("alpha", 0.35))
            
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
        np.array([np.array(polygon.exterior.xy[0]).min(), np.array(polygon_b.exterior.xy[0]).min()]).min() - lim_buffer, 
        np.array([np.array(polygon.exterior.xy[0]).max(), np.array(polygon_b.exterior.xy[0]).max()]).max() + lim_buffer, 
    )
    axs[0].set_ylim(
        np.array([np.array(polygon.exterior.xy[1]).min(), np.array(polygon_b.exterior.xy[1]).min()]).min() - lim_buffer, 
        np.array([np.array(polygon.exterior.xy[1]).max(), np.array(polygon_b.exterior.xy[1]).max()]).max() + lim_buffer, 
    )
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xlim(
        np.array([np.array(polygon.exterior.xy[0]).min(), np.array(polygon_b.exterior.xy[0]).min()]).min() - lim_buffer, 
        np.array([np.array(polygon.exterior.xy[0]).max(), np.array(polygon_b.exterior.xy[0]).max()]).max() + lim_buffer, 
    )
    axs[1].set_ylim(
        np.array([np.array(polygon.exterior.xy[1]).min(), np.array(polygon_b.exterior.xy[1]).min()]).min() - lim_buffer, 
        np.array([np.array(polygon.exterior.xy[1]).max(), np.array(polygon_b.exterior.xy[1]).max()]).max() + lim_buffer, 
    )
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plot_fibers(fibers, axs[0])
    clipped: np.ndarray = jax.vmap(clip_to_polygon, in_axes=(0, None))(fibers, polygon)
    clipped: np.ndarray = jax.vmap(clip_to_polygon, in_axes=(0, None))(clipped, polygon_b)
    clipped = clipped[np.all(clipped[:, 1] != clipped[:, 0], axis=-1), ...]
    plot_fibers(clipped, axs[1], body_color="tab:red", endpoints=True, alpha=1.)
    plt.savefig(save_path)
    plt.close()

def main(): 
    vertices: np.ndarray = np.array([
        [0., 0.], 
        [0.5, 0.3], 
        [1., 0.], 
        [0.7, 0.5], 
        [1., 1.], 
        [0.5, 0.7], 
        [0., 1.], 
        [0.3, 0.5], 
    ])
    vertices_b: np.ndarray = np.array([
        [0.5, 0.5], 
        [1., 0.5], 
        [1., 1.], 
        [0.5, 1.], 
    ])
    polygon = shapely.Polygon(vertices)
    polygon_b = shapely.Polygon(vertices_b)

    # sample fibers 
    key: np.ndarray = npr.PRNGKey(0)
    boundary_buffer: float = 0.
    bounds: np.ndarray = np.array([0. - boundary_buffer, 0. - boundary_buffer, 1. + boundary_buffer, 1. + boundary_buffer])
    num_fibers: int = 350
    fiber_length: float = 0.2 
    fibers: np.ndarray = estimators.sample(key, bounds, num_fibers, fiber_length)

    polygons: tuple[shapely.Polygon] = (polygon, polygon_b)
    save_path: Path = Path("polygon_intersection.png")
    show(polygons, fibers, save_path)
    print(f"Saved demo figure to: {str(save_path)}")




if __name__=="__main__":
    main()