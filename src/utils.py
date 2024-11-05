import functools 

import jax 
import jax.numpy as np

# @functools.partial(jax.jit, static_argnums=(1,))
def custom_norm(x: np.ndarray, dtype: type=np.float32) -> np.ndarray: 
    """Utility function for computing the 2-norm of an array `x` 
    in a way that is safe under differentiation/tracing. 

    Parameters
    ----------
    x : np.ndarray
        Array for which to compute the 2-norm. 
    dtype : jax._src.numpy.lax_numpy._ScalarMeta
        Numeric type of the returned zero element (in the case where 
        the given array has norm zero). 

    Returns
    -------
    norm : np.ndarray [,]
        2-norm of x. 
    """
    is_zero: np.ndarray = np.allclose(x, 0.)
    x: np.ndarray = np.where(is_zero, np.ones_like(x), x)
    return np.where(is_zero, 0., np.linalg.norm(x))
    
def divide00(numerator: np.ndarray, denominator: np.ndarray, dtype: type=np.float32) -> np.ndarray:
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
        numeric type of the result (default np.float32)

    Returns 
    -------
    quotient : np.ndarray 
        autodiff safe quotient. 
    """
    force_zero: np.ndarray = np.logical_and(numerator == 0, denominator == 0)
    quotient: np.ndarray = np.where(force_zero, dtype(0.0), numerator) / np.where(force_zero, dtype(1.0), denominator)
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