import numpy as np
import numba as nb

@nb.jit(cache = True, nopython = True)
def constanth(m,i,h_args):
    Nx, Ny, strength,dir = h_args
    h = np.zeros_like(m)
    h[:,:,int(dir)] = strength
    return h