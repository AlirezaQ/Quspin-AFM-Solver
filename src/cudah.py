from re import S
import numpy as np
import numba as nb
import numba.cuda as cuda
from numba import float64
from numba.cuda import random as rd
import cmath as cm
from math import sqrt
import sys


@cuda.jit(device =True)
def h(m,h,timestep,dim):
    #start_i,end_i,strength,dir_h
    start_i = h[0]
    end_i = h[1]
    dir_h = h[3]
    strength = h[2]
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    j = cuda.blockDim.y*cuda.blockIdx.y + cuda.threadIdx.y
    if dim ==dir_h and j>start_i and j <end_i:
        return strength
    return 0