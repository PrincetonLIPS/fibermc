from typing import Optional 

import jax 
import jax.numpy as np 
import jax.random as npr 
import numpy as onp 
import pytest
import shapely 
import torch

import src.fibermc.estimators as jax_estimators
import src.fibermc.geometry_utils as jax_geom_utils
import torch_src.estimators as torch_estimators 
import torch_src.geometry_utils as torch_geom_utils 

jax.config.update("jax_disable_jit", True)

key: np.ndarray = npr.PRNGKey(0)
DTYPE: type = np.float32
DOMAIN_BOUNDS: np.ndarray = np.array([0., 0., 1., 1.])
FIBER_LENGTH: np.ndarray = np.array(2e-01)
NUM_FIBERS: int = 25

@pytest.fixture
def random_convex_polygon() -> tuple[shapely.Polygon, np.ndarray]: 
    num_points: int = 7 
    xkey, ykey = npr.split(key)
    points: np.ndarray = np.stack(
    (
            npr.uniform(xkey, (num_points,), minval=DOMAIN_BOUNDS[0], maxval=DOMAIN_BOUNDS[2]), 
            npr.uniform(ykey, (num_points,), minval=DOMAIN_BOUNDS[1], maxval=DOMAIN_BOUNDS[3])
    ), axis=-1)
    convex_hull: shapely.Polygon = shapely.convex_hull(shapely.MultiPoint(points))
    convex_hull_points: np.ndarray = np.array(convex_hull.exterior.coords)[1:]
    return (convex_hull, convex_hull_points)


def test_clip_forward(random_convex_polygon): 
    _, vertices = random_convex_polygon 
    fiber_key, _ = npr.split(key)
    fibers: np.ndarray = jax_estimators.sample(fiber_key, DOMAIN_BOUNDS, NUM_FIBERS, FIBER_LENGTH, dtype=DTYPE)
    clipped_fibers: np.ndarray = jax_geom_utils.clip_inside_convex_hull(fibers, vertices[:-1])
    fibers_torch = torch.from_numpy(onp.array(fibers))
    vertices_torch = torch.from_numpy(onp.array(vertices[:-1]))
    torch_clipped = torch_geom_utils.polygon_clip(fibers_torch, vertices_torch)
    assert onp.allclose(onp.array(clipped_fibers), torch_clipped.numpy())


def test_clip_backward(): 
    pass 

def test_implicit_clip_forward(): 
    pass 

def test_implicit_clip_backward(): 
    pass 