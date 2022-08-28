import numpy as np
import numba as nb
import numba.cuda as cuda
from numba import float64
from numba.cuda import random as rd
import cmath as cm
from math import sqrt
import sys

@cuda.jit(device=True)
def J(m,J,timestep,geom,dim):
    from_ind = int(J[0])
    to_ind = int(J[1])
    dir = int(J[2])
    strength = J[3]
    bias_strength = J[4]
    timestep_stop = J[5]

    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    j = cuda.blockDim.y*cuda.blockIdx.y + cuda.threadIdx.y
    if geom[i,j,0] != 0:
        if j>= from_ind and j<to_ind and timestep < timestep_stop:
            if dim == dir:
                return strength
        if j>110 and j<140:
            if dim == dir:
                return bias_strength
    """if geom[i,j,0] != 0 and timestep > 300:
        if j>= from_ind and j<to_ind:
            if dim == dir:
                return 0.1*strength"""
    return 0.0 