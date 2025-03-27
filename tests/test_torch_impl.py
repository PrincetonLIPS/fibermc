from typing import Optional 

import jax 
import jax.numpy as np 
import jax.random as npr 
import numpy as onp 
import pytest
import shapely 
import torch
from torch import Tensor
import torch.nn as nn 

import src.fibermc.estimators as jax_estimators
import src.fibermc.geometry_utils as jax_geom_utils
import src.fibermc.implicit_differentiation as jax_implicit_diff
import torch_src.estimators as torch_estimators 
import torch_src.geometry_utils as torch_geom_utils 
import torch_src.implicit_differentiation as torch_implicit_diff

jax.config.update("jax_disable_jit", True)
torch.manual_seed(0)

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


def test_clip_backward(random_convex_polygon): 
    _, vertices = random_convex_polygon 
    fiber_key, _ = npr.split(key)
    fibers: np.ndarray = jax_estimators.sample(fiber_key, DOMAIN_BOUNDS, NUM_FIBERS, FIBER_LENGTH, dtype=DTYPE)
    fibers_torch = torch.from_numpy(onp.array(fibers)).requires_grad_(True)
    vertices_torch = torch.from_numpy(onp.array(vertices[:-1]))

    def jax_obj(fibers: np.ndarray) -> np.ndarray: 
        clipped = jax_geom_utils.clip_inside_convex_hull(fibers, vertices[:-1])
        return clipped.sum()**2

    def torch_obj(fibers):
        clipped = torch_geom_utils.polygon_clip(fibers, vertices_torch)
        return clipped.sum()**2

    jax_gradient = jax.grad(jax_obj)(fibers)
    out = torch_obj(fibers_torch)
    out.backward()
    torch_gradient = fibers_torch.grad 
    assert onp.allclose(onp.array(jax_gradient), torch_gradient.numpy(), atol=1e-03)


def test_implicit_clip_forward(): 
    class TorchImplicitModule(nn.Module):
        def __init__(self): 
            super().__init__()
            self.fc = nn.Linear(2, 1, bias=False)

        def forward(self, x: Tensor) -> Tensor: 
            return self.fc(x)**2

    torch_implicit = TorchImplicitModule()
    implicit_layer = torch_implicit_diff.BisectionLayer(torch_implicit)

    fiber_key, _ = npr.split(key)
    fibers: np.ndarray = jax_estimators.sample(fiber_key, DOMAIN_BOUNDS, NUM_FIBERS, FIBER_LENGTH, dtype=DTYPE)
    fibers_torch = torch.from_numpy(onp.array(fibers)).requires_grad_(True)

    # compute fixed point (intersections)
    out = implicit_layer(fibers_torch)

    # compute intersection over union 
    z = out.sum()**2 
    z.backward()


    # jax side 
    def jax_f(params, x): 
        A = params[0]
        return (A @ x).sum()**2

    params = (np.array(torch_implicit.fc.weight.data.numpy()),)

    def jax_fwd(params): 
        jax_out = jax.vmap(lambda fiber: jax_implicit_diff.bisection_solver(params, fiber, jax_f))(fibers)
        return jax_out.sum()**2

    jax_grad = jax.grad(jax_fwd)(params)[0]

    torch_grad = -torch_implicit.fc.weight.grad.numpy() # we populate this buffer as the negative gradient for optimizers
    jax_grad = onp.array(jax_grad)

    assert onp.allclose(torch_grad, jax_grad)










def test_implicit_clip_backward(): 
    pass 