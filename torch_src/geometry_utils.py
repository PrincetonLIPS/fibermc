from typing import Optional


from jaxtyping import Float, Bool
import torch 
import torch.nn as nn 
from torch import Tensor 

def segment(vertices: Float[Tensor, "V 2"]) -> Float[Tensor, "V-1 2 2"]:
    return torch.stack([vertices, torch.roll(vertices, -1, dims=-2)], dim=-2)

def in_unit_interval(x: Float[Tensor, "..."], open: Optional[bool]=True) -> Bool[Tensor, "..."]: 
    return torch.logical_and(0 < x, x < 1) if open else torch.logical_and(0 <= x, x <= 1)

def segment_intersection(x: Float[Tensor, "M 2 2"], y: Float[Tensor, "N 2 2"]) -> Float[Tensor, "M N 2"]: 
    M, N = x.shape[0], y.shape[0] 
    A: Float[Tensor, "M N 2 2"] = torch.stack((
        (x[:, 1] - x[:, 0])[:, torch.newaxis, :].expand(-1, N, -1), 
        (y[:, 0] - y[:, 1])[torch.newaxis, ...].expand(M, -1, -1)
    ), dim=-1)
    b: Float[Tensor, "M N 2"] = y[torch.newaxis, :, 0] - x[:, torch.newaxis, 0]
    abs_determinant: Float[Tensor, "M N"] = torch.abs(torch.linalg.det(A))
    solutions: Float[Tensor, "M N 2"] = torch.linalg.solve(A, b)
    no_intersection: Float[Tensor, "M N 2"] = torch.tensor([torch.inf, torch.inf]).expand_as(solutions)
    return torch.where(abs_determinant[..., torch.newaxis] > 0, solutions, no_intersection)

def orient_polygon(vertices: Float[Tensor, "V 2"]) -> Float[Tensor, "V 2"]: 
    # TODO add batch dimension
    pad: callable = lambda x: nn.functional.pad(x, (0, 1))
    orientation: float = torch.sign(torch.cross(pad(vertices[1] - vertices[0]), pad(vertices[2] - vertices[0]))[-1]).item()
    if (orientation == -1.): 
        return torch.flip(vertices, (0,))
    else: 
        return vertices 

def _clip_against_segments(fibers: Float[Tensor, "F 2 2"], segments: Float[Tensor, "N 2 2"]) -> tuple[Tensor]: 
    F, N = fibers.shape[0], segments.shape[0] 
    lhs = nn.functional.pad((segments[:, 1] - segments[:, 0]).expand(F, -1, -1), (0, 1)).view(-1, 3)
    endpoint_sides: Float[Tensor, "F N 2"] = torch.stack((
        torch.sign(torch.linalg.cross(lhs, nn.functional.pad(fibers[:, torch.newaxis, 0] - segments[torch.newaxis, :, 0], (0, 1)).view(-1, 3))[:, -1]), 
        torch.sign(torch.linalg.cross(lhs, nn.functional.pad(fibers[:, torch.newaxis, 1] - segments[torch.newaxis, :, 0], (0, 1)).view(-1, 3))[:, -1]), 
    ), dim=-1).view(F, N, 2)
    start_sides, end_sides = endpoint_sides[..., 0], endpoint_sides[..., 1]

    # find the intersection (if it exists)
    intersection: Float[Tensor, "F N 2"] = segment_intersection(fibers, segments)
    has_intersection: Bool[Tensor, "F N"] = in_unit_interval(intersection, open=True).all(-1)

    # for a fiber `[x0, x1]` we have intersections at `(1-a)x0 + (a)x1`
    alpha: Float[Tensor, "F N"] = intersection[..., 0]

    intersection_select: Float[Tensor, "F N 2"] = torch.where(
        (start_sides > end_sides)[..., torch.newaxis],
        torch.stack((torch.zeros_like(alpha), alpha), dim=-1),
        torch.stack((alpha, torch.ones_like(alpha)), dim=-1),
    )

    # determine the clip parameters
    clip_parameters: Float[Tensor, "F N 2"] = torch.where(
        has_intersection[..., torch.newaxis], intersection_select, torch.tensor([0.0, 1.0]).expand_as(intersection_select)
    )
    return clip_parameters, endpoint_sides, has_intersection

def polygon_clip(fibers: Float[Tensor, "F 2 2"], vertices: Float[Tensor, "V 2 2"]) -> Float[Tensor, "F 2 2"]:
    # TODO polygon orientation
    vertices: Float[Tensor, "V 2 2"] = orient_polygon(vertices)
    segments: Float[Tensor, "N 2 2"] = segment(vertices)

    # clip the fibers with respect to the segments
    clips, endpoint_sides, has_intersections = _clip_against_segments(fibers, segments)

    lo, hi = clips[..., 0], clips[..., 1]
    clip_params: Float[Tensor, "F 2"] = torch.stack([lo.max(1)[0], hi.min(1)[0]], axis=-1)

    # determine if there are any intersections
    any_intersections: Float[Tensor, "F"] = has_intersections.any(-1)

    is_fully_inside: Tensor = (endpoint_sides > 0).all((-1, -2))
    is_fully_outside: Tensor = torch.logical_and(~is_fully_inside, ~any_intersections)

    # compute the clipping parameters
    clip_parameters: Tensor = torch.where(
        is_fully_outside[:, torch.newaxis], torch.zeros_like(clip_params), clip_params
    )

    x0, x1 = fibers[:, 0], fibers[:, 1]
    clipped: Tensor = ((1 - clip_parameters[:, torch.newaxis, :]) * x0[..., torch.newaxis] + clip_parameters[:, torch.newaxis, :] * x1[..., torch.newaxis]).transpose(-1, -2)

    return clipped



