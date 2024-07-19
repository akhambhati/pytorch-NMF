"""
Simple tensor operations and utility functions.

Some of these functions were ported with minor modifications
from the tensorly package, https://tensorly.github.io/, distributed
under a BSD clause 3 license.
"""
import numpy as np


def unfold(tensor, mode):
    """Returns the mode-`mode` unfolding of `tensor`.

    Parameters
    ----------
    tensor : ndarray
    mode : int

    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``

    Author
    ------
    Jean Kossaifi <https://github.com/tensorly>
    """
    return np.moveaxis(tensor, mode, 0).reshape((tensor.shape[mode], -1))

def ein_outer(n_modes):
    chars = [chr(97+i) for i in range(n_modes)]
    estr = '...,'.join(chars)+'...'+'->'+''.join(chars)
    return estr

def outer_prod(matrices):
    return torch.einsum(ein_outer(len(matrices)), matrices)

def khatri_rao(matrices):
    outprod = outer_prod(matrices)
    return outprod.reshape(-1, outprod.shape[-1]) 
