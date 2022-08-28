import numpy as np
import numba as nb

@nb.jit(cache = True, nopython = True, fastmath = True)
def constantJ(m,i,J_args):
    Nx, Ny, strength,dir = J_args
    J = np.zeros_like(m)
    J[:,:,int(dir)] = strength
    return J

@nb.jit(cache = True, nopython = True, fastmath = True)
def constantJinArea(m,i,J_args):
    from_ind, to_ind,dir_J, strength = J_args
    J = np.zeros_like(m)
    J[:,from_ind:to_ind, int(dir_J)] = strength
    return J

@nb.jit(cache = True, nopython = True, fastmath = True)
def Jpulses(m,i,J_args):
    return 0