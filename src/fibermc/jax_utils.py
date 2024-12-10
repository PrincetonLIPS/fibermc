import functools 
from typing import List, TypeVar

import numpy as static_np
import jax 
import jax.numpy as np 
import jax.tree_util as tree_util
import optax

pytree: type = TypeVar("Pytree")
FP32: type = np.float32
FP64: type = np.float64

def merge_trees(trees: List[pytree]) -> pytree: 
    merged: pytree = trees[0]
    for tree in trees[1:]: 
        merged: pytree = tree_util.tree_map(lambda t, a: np.vstack((t, a)), merged, tree)
    return merged

def index_trees(trees: pytree, index: int) -> pytree: 
    return tree_util.tree_map(lambda x: x[index], trees)

def divide_pytree(tree: pytree, divisor: float) -> pytree:
  return tree_util.tree_map(lambda pt: pt / divisor, tree)

def add_pytrees(first_pytree: pytree, second_pytree: pytree) -> pytree:
  return tree_util.tree_map(lambda first_tree, second_tree: first_tree + second_tree, first_pytree, second_pytree)

def tree_stack(trees: List[pytree]):
        """Takes a list of trees and stacks every corresponding leaf.
        For example, given two trees ((a, b), c) and ((a', b'), c'), returns
        ((stack(a, a'), stack(b, b')), stack(c, c')).
        Useful for turning a list of objects into something you can feed to a
        vmapped function.
        """
        leaves_list = []
        treedef_list = []
        for tree in trees:
            leaves, treedef = tree_util.tree_flatten(tree)
            leaves_list.append(leaves)
            treedef_list.append(treedef)

        grouped_leaves = zip(*leaves_list)
        result_leaves = [np.stack(l) for l in grouped_leaves]
        return treedef_list[0].unflatten(result_leaves)

def divides(a: int, b: int) -> bool: 
    return a % b == 0 

def array_size(arr: np.ndarray) -> int:
    # TODO unnecessary
    return static_np.array(arr).nbytes

def _vectorize(signature: str, excluded: tuple=()) -> callable: 
    def decorator(f: callable) -> callable:
        vectorized: callable = np.vectorize(f, excluded=excluded, signature=signature)
        return vectorized
    return decorator

def _jit_vectorize(signature: str, excluded: tuple =()) -> callable:
    """Applies the jax.jit and jax.numpy.vectorize transformations 
    to the tagged function.
    """
    def decorator(f: callable) -> callable:
        vectorized: callable = np.vectorize(f, excluded=excluded, signature=signature)
        jitted_and_vectorized: callable = jax.jit(vectorized, static_argnums=excluded)
        return jitted_and_vectorized
    return decorator

#    squared_sum: float = x.dot(x)
#    is_zero: callable = lambda _: numeric_type(0) 
#    is_nonzero: callable = lambda x: np.sqrt(x) 
#    norm: float = jax.lax.cond(squared_sum == 0, is_zero, is_nonzero, operand=squared_sum)
#    return norm 

def divide00(numerator: np.ndarray, denominator: np.ndarray, numeric_type: type=FP32) -> np.ndarray:
    """Computes the quotient of the given numerator and denominator 
    such that zero divided by zero equals zero, and differentiation 
    works. 

    Parameters
    ----------
    numerator : np.ndarray 
        numerator of the quotient 
    denominator : np.ndarray 
        denominator of the quotient 
    numeric_type : type 
        numeric type of the result (default FP32)

    Returns 
    -------
    quotient : np.ndarray 
        autodiff safe quotient. 
    """
    force_zero: np.ndarray = np.logical_and(numerator == 0, denominator == 0)
    quotient: np.ndarray = np.where(force_zero, numeric_type(0.0), numerator) / np.where(force_zero, numeric_type(1.0), denominator)
    return quotient 

def zero_one_sign(arr: np.ndarray) -> np.ndarray:
    """Returns an array of the same shape as the input with 
    the value 1. where the input array is greater than or equal 
    to zero and 0. where the input array is less than zero. 

    Parameters
    ----------
    arr : np.ndarray 
        input array 

    Returns 
    -------
    binary_arr : np.ndarray 
        result with shape of `arr` and value 1. where arr >= 0. 
        and value 0. otherwise. 
    """
    binary_arr: np.ndarray = 0.5 * (1.0 + np.sign(arr))
    return binary_arr

def vectorized_cond(predicate, true_function, false_function, operand) -> np.ndarray:
    # true_fun and false_fun must act elementwise (i.e. be vectorized)
    true_op = np.where(predicate, operand, 0)
    false_op = np.where(predicate, 0, operand)
    return np.where(predicate, true_function(true_op), false_function(false_op))
