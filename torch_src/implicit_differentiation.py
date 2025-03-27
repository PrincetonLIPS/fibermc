import functools
from typing import Callable, Optional

from jaxtyping import Float 
import torch 
from torch import Tensor
import torch.nn as nn 
import torch.utils._pytree as pytree

def bisect(f: callable, fibers: Float[Tensor, "F 2 2"], num_iterations: Optional[int] = 10) -> Float:
    # TODO make sure batching works correctly
    interpolant: Callable[[Tensor], Tensor] = lambda x: fibers[:, 0] + x * (fibers[:, 1] - fibers[:, 0])
    h: Callable[[Tensor], Tensor] = lambda x: f(interpolant(x))

    # standardize so the 'left' endpoint has negative value
    F: int = fibers.shape[0]
    endpoints: Float[Tensor, "F 2"] = torch.where(h(0.) > 0, torch.tensor([1., 0.]).expand(F, -1), torch.tensor([0., 1.]).expand(F, -1))

    def _bisect(endpoints: Tensor) -> Tensor:
        left, right = endpoints[..., 0], endpoints[..., 1]
        midpoint: Tensor = (left + right) / 2.
        return torch.where((h(midpoint[:, torch.newaxis]) < 0.), torch.stack((midpoint, right), dim=-1), torch.stack((left, midpoint), dim=-1))

    for _ in range(num_iterations):
        endpoints: Tensor = _bisect(endpoints)

    return endpoints[:, 0]

def get_interpolant(alpha: Tensor, fibers: Tensor) -> Tensor:
    return fibers[:, 0] + alpha * (fibers[:, 1] - fibers[:, 0])

def bisection_constraint(
    f: callable, x: Tensor, params: tuple[Tensor], fibers: Tensor
    ) -> Tensor:
    z: Tensor = get_interpolant(x, fibers)
    field_constraint: Tensor = torch.squeeze(f(z, params))
    return field_constraint

class BisectionLayer(nn.Module):
    def __init__(self, f: callable): 
        super().__init__()
        self.implicit_fn = f
        _, self.param_spec = pytree.tree_flatten({**dict(self.implicit_fn.named_parameters(), **dict(self.implicit_fn.named_buffers()))})

        def implicit_functional(x: Tensor, *params: tuple[Tensor]) -> Tensor: 
            structured_params = pytree.tree_unflatten(*params, self.param_spec)
            return torch.func.functional_call(self.implicit_fn, structured_params, (x,))

        self.implicit_functional = implicit_functional

        class BisectionSolver(torch.autograd.Function):
            @staticmethod
            def forward(ctx, fibers: Tensor):
                _params, _ = pytree.tree_flatten({**dict(self.implicit_fn.named_parameters(), **dict(self.implicit_fn.named_buffers()))})
                fixed_point: Tensor = bisect(lambda x: self.implicit_functional(x, _params), fibers)[:, torch.newaxis]
                ctx.save_for_backward(fixed_point, fibers)
                return fixed_point


            @staticmethod
            def backward(
                ctx, incoming_gradient: Float[Tensor, "F 1"],
            ) -> tuple[Tensor]:
                # unpack residuals
                fixed_point, fibers = ctx.saved_tensors
                _params, _ = pytree.tree_flatten({**dict(self.implicit_fn.named_parameters(), **dict(self.implicit_fn.named_buffers()))})

                # f's univariate analogues
                f_params: callable = lambda params: bisection_constraint(
                    self.implicit_functional, fixed_point, params, fibers
                )
                f_spatial: callable = lambda _x: bisection_constraint(self.implicit_functional, _x, _params, fibers)

                # partial vjps (w.r.t. params and the spatial variable)
                _, vjp_params = torch.func.vjp(f_params, _params)
                _, vjp_spatial = torch.func.vjp(f_spatial, fixed_point)

                # solve for the intermediate vjp
                jacobian_f_fn: callable = torch.func.jacrev(lambda x: f_spatial(x).sum(0))
                jacobian_f: Float[Tensor, "F 1"] = jacobian_f_fn(fixed_point)

                # ensure A > 0 
                intermediate_vjp: Float[Tensor, "F 1"] = -1.0 * (incoming_gradient / jacobian_f)
                final_vjp: Tensor = vjp_params(intermediate_vjp.squeeze(-1))
                gradient_tree = pytree.tree_unflatten(final_vjp, self.param_spec)

                for name, parameter in self.implicit_fn.named_parameters(): 
                    parameter.grad = -gradient_tree[name][0] # TODO why is this a list? 

                return (None,)

        self._fwd = BisectionSolver.apply

    def forward(self, fibers: Tensor) -> Tensor: 
        # TODO forward should be a fn of implicit fn params 
        return self._fwd(fibers)
