from numba.cuda.cudadrv.driver import Device
import numpy as np
import numba as nb
import time
import random
from tqdm import tqdm
import plotting as plot
import utilities as ut
import config
import matplotlib.pyplot as plt
import numba.cuda as cuda
import RK4 as RK4
from numba import float64
from numba.cuda import random as rd
import cmath as cm
from math import sqrt
import sys

temp_shape = (15,3)
@cuda.jit(device = True)
def normalize1D(m,dim):
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    norm = np.sqrt(m[i,0]**2 + m[i,1]**2 + m[i,2]**2)
    return m[i,dim]/norm

@cuda.jit(device = True)
def normalize2D(m,geom,dim):
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    j = cuda.blockDim.y*cuda.blockIdx.y + cuda.threadIdx.y
    norm = np.sqrt(m[i,j,0]**2 + m[i,j,1]**2 + m[i,j,2]**2)
    if geom[i,j,0] !=0:
        return m[i,j,dim]/norm
    else:
        return 0

@cuda.jit(device = True)
def cudaCrossProduct1D(v1, v2,dim):
    #To invoke call cudaCrossProduct[blockspergrid, threadsperblock](A)
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    if i >= 0 and i < v1.shape[0]:
        if dim ==0:
            return v1[i,1]*v2[i,2] - v1[i,2]*v2[i,1]
        if dim ==1:
            return v1[i,2]*v2[i,0] - v1[i,0]*v2[i,2]
        if dim == 2:
            return v1[i,0]*v2[i,1] - v1[i,1]*v2[i,0]
    
@cuda.jit(device= True)
def scalarMatrixMul(A,c):
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    if i>0 and i < A.shape[0]:
        A[i,0] = c*A[i,0]
        A[i,1] = c*A[i,1]
        A[i,1] = c*A[i,1]
    return A

@cuda.jit(device = True)
def matrixSubstract(A,B):
    """Substracts B from A
    """
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    if i>0 and i < A.shape[0]:
        A[i,0] = A[i,0] - B[i,0]
        A[i,1] = A[i,1] - B[i,1]
        A[i,2] = A[i,2] - B[i,2]
    return A

@cuda.jit(device = True)
def cudaLaplacian1D(mesh, border_size, dim):
    """Calculates the laplacian in 1D using finite difference on 5 points. This is needed for the exchange energy-term of the hamiltonian. 

    Args:
        mesh (np.ndarray): The input is a time-part of the spins-matrix, so mesh is of size (N,3), ie N spins with all 3 components. 

    Returns:
        [np.array]: Returns the laplacian of the mesh-matrix. It is of the same dimension as the mesh-matrix. 
    """
    const = 1/(12)
    N = mesh.shape[0]
    result= 0
    #Laplacian middle values
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    if i >= 2 and i <N-2 and i >= border_size and i < N-border_size:
        return (const)*(-mesh[i-2,dim] +16*mesh[i-1,dim] -30*mesh[i, dim] + 16*mesh[i+1,dim] - mesh[i+2,dim])
    
    const2 = 12/25
    
    #Laplacian edge values(same as Alireza)
    if i == 0 and i >= border_size:
        return const2*(4*mesh[i+1,dim] - 3*mesh[i+2,dim] + (4/3)*mesh[i+3,dim] - 0.25*mesh[i+4,dim])
    #laplacian[N-1,:] = const2*(4*mesh[N-2,:]  - 3*mesh[N-3,:] + (4/3)*mesh[N-4,:] -0.25*mesh[N-5,:])
    
    
    #Next to edge (Same as Alireza)
    if i == 1 and i >= border_size: 
        return const*(10*mesh[i-1,dim] - 15*mesh[i,dim] - 4*mesh[i+1,dim] + 14*mesh[i+2,dim] - 6*mesh[i+3,dim] + 1*mesh[i+4,dim])
        
    #laplacian[N-2,:] = const*(10*mesh[N-1,:] - 15*mesh[N-2,:] - 4*mesh[N-3,:] +14*mesh[N-3,:] -6*mesh[N-4,:] + mesh[N-5,:])
    
    """
    laplacian[0,:] = -(3*mesh[0,:] - 4*mesh[1,:] + mesh[2,:])/2
    laplacian[1,:] = -(3*mesh[1,:] - 4*mesh[2,:] + mesh[3,:])/2
    """
    if i == N-3 and i < N-border_size:
        return -(3*mesh[i,dim] - 4*mesh[i-1,dim] + mesh[i-2,dim])/2
    if i == N-2 and i < N-border_size:
        return -(3*mesh[i,dim] - 4*mesh[i-1,dim] + mesh[i-2,dim])/2
    
@cuda.jit(device=True)
def cudaExchange2(m_1, m_2, w_ex, A_1, A_2, border_size, dim):
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    return -(w_ex*m_2[i,dim] -A_1*cudaLaplacian1D(m_1, border_size, dim) - A_2*cudaLaplacian1D(m_2, border_size, dim))
    
@cuda.jit(device = True)
def cudaZeeman(i,h,border_size,dim):
    j = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    return h[j,i,dim]

@cuda.jit(device=True)
def cudaAnisotropy1D(m_a,m_b,w_1,w_2,dim):
    """Updates the  anisotropy of the spin located on the lattice i

    Args:
        i ([int]): [thread idex from GPU]
        m_a ([Nx,3]): [a matrix of]
        m_b ([Nx,3]): [description]
        w_1 ([3,3]): [description]
        w_2 ([3,3]): [description]
        result ([Nx,3]): [description]
    """
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    if dim == 0:
        return -1* w_1[i,0,0]*m_a[i,0] - w_1[i,0,1]*m_a[i,1] - w_1[i,0,2]*m_a[i,2] - w_2[i,0,0]*m_b[i,0] - w_2[i,0,1]*m_b[i,1] - w_2[i,0,2]*m_b[i,2]
    elif dim == 1:
        return-1* w_1[i,1,0]*m_a[i,0] - w_1[i,1,1]*m_a[i,1] - w_1[i,1,2]*m_a[i,2] - w_2[i,1,0]*m_b[i,0] - w_2[i,1,1]*m_b[i,1] - w_2[i,1,2]*m_b[i,2]
    else:
        return -1* w_1[i,2,0]*m_a[i,0] - w_1[i,2,1]*m_a[i,1] - w_1[i,2,2]*m_a[i,2] - w_2[i,2,0]*m_b[i,0] - w_2[i,2,1]*m_b[i,1] - w_2[i,2,2]*m_b[i,2]

@cuda.jit(device = True)
def cudaAnisotropy2D(m_a, m_b, w_1,w_2, dim):
    """Updates the  anisotropy of the spin located on the lattice i,j

    Args:
        i ([type]): [description]
        j ([type]): [description]
        m_a ([type]): [description]
        m_b ([type]): [description]
        w_1 ([type]): [description]
        w_2 ([type]): [description]
        result ([type]): [description]
    """
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    j = cuda.blockDim.y*cuda.blockIdx.y + cuda.threadIdx.y
    if dim == 0:
        return -1* w_1[i,j,0,0]*m_a[i,j,0] - w_1[i,j,0,1]*m_a[i,j,1] - w_1[i,j,0,2]*m_a[i,j,2] - w_2[i,j,0,0]*m_b[i,j,0] - w_2[i,j,0,1]*m_b[i,j,1] - w_2[i,j,0,2]*m_b[i,j,2]
    elif dim == 1:
        return -1* w_1[i,j,1,0]*m_a[i,j,0] - w_1[i,j,1,1]*m_a[i,j,1] - w_1[i,j,1,2]*m_a[i,j,2] - w_2[i,j,1,0]*m_b[i,j,0] - w_2[i,j,1,1]*m_b[i,j,1] - w_2[i,j,1,2]*m_b[i,j,2]
    else: 
        return -1* w_1[i,j,2,0]*m_a[i,j,0] - w_1[i,j,2,1]*m_a[i,j,1] - w_1[i,j,2,2]*m_a[i,j,2] - w_2[i,j,2,0]*m_b[i,j,0] - w_2[i,j,2,1]*m_b[i,j,1] - w_2[i,j,2,2]*m_b[i,j,2]

@cuda.jit(device = True)
def cudaDMIEnergy1D(m,a,d,dim):
    """Calculates the DMI term given by m x d. However, depending on m_a or m_b, 
    this term differns, and therefore we include the variable "a" to differentiate between them.

    Args:
        m ([Nx,3]): [either m_a or m_b]
        i ([int]): [thread index]
        is_a ([bool]): [If true, the m-matrix is m_a. if false, it is m_b]
        result ([type]): [the  return matrix. This  needs to be changed in the GPU implementation]
    """
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    if a:
        if dim == 0:
            return -(m[i,1]*d[2] - m[i,2]*d[1]) 
        elif dim == 1:
            return -(-m[i,0]*d[2] + m[i,2]*d[0])
        else:
            return -(m[i,0]*d[1] - m[i,1]*d[0])
    else:
        if dim ==0:
            return (m[i,1]*d[2] - m[i,2]*d[1])
        elif dim ==1:
            return (m[i,0]*d[2] - m[i,2]*d[0])
        else:
            return (m[i,0]*d[1]- m[i,1]*d[0])

@cuda.jit(device = True)
def cudaDMIEnergy2D(m,is_a,d,dim):
    """Calculates the DMI term given by m x d. However, depending on m_a or m_b, 
    this term differns, and therefore we include the variable "a" to differentiate between them.

    Args:
        m ([Nx,Ny, 3]): [either m_a or m_b]
        i ([int]): [thread index]
        is_a ([bool]): [If true, the m-matrix is m_a. if false, it is m_b]
    """
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    j = cuda.blockDim.y*cuda.blockIdx.y + cuda.threadIdx.y
    if is_a: #This is the a matrix
        if dim == 0:
            return -(m[i,j,1]*d[2] - m[i,j,2]*d[1]) 
        elif dim == 1:
            return -(-m[i,j,0]*d[2] + m[i,j,2]*d[0])
        else:
            return -(m[i,j,0]*d[1] - m[i,j,1]*d[0])
    else: #If not a, it is the b matrix
        if dim ==0:
            return (m[i,j,1]*d[2] - m[i,j,2]*d[1])
        elif dim ==1:
            return (m[i,j,0]*d[2] - m[i,j,2]*d[0])
        else:
            return (m[i,j,0]*d[1]- m[i,j,1]*d[0])

@cuda.jit(device=True)
def CudaSecondDMI1D(m,dim):
    """Performs the cross-product-part of the DMI calculations using dx = dy = dz = 1 due to the non-dimensionality of the equations
    This is currently written using the central stencil using a point to the left and right of the point in question.

    Args:
        m ([np.ndarray(Nx,3)]): [The entire grid of spins at time i]
        border_size ([int], optional): [The size of the borders where the spins are fixed]. Defaults to border_size.

    Returns:
        [type]: [description]
    """
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    if dim == 0:
        return 0
    elif dim == 1:
        if i >= 2 or i <= m.shape[0]-3:
            return (1/12)*(m[i-2,2] -8*m[i-1,2] +8*m[i+1,2] - m[i+2,2])
        if i == 0:
            return (1/12)*(-25*m[i,2] + 48*m[i+1,2] - 36*m[i+2,2] + 16*m[i+3,2] - 3*m[i+4,2])
            
        if i == 1:
            return (1/12)*(-3*m[i-1,2] - 10*m[i,2] + 18*m[i+1,2] - 6*m[i+3,2] + m[i+4,2])
            
        if i == m.shape[0]-2:
            return (1/12)*(-m[i-3,2] + 6*m[i-2,2] - 18*m[i-1,2] + 10*m[i,2] + 3*m[i+1,2])
            
        if i == m.shape[0]-1:
            return (1/12)*(3*m[i-4,2] - 16*m[i-3,2] + 36*m[i-2,2] - 48*m[i-1,2] + 25*m[i,2])
            
    else:
        if i >= 2 or i <= m.shape[0]-3:
            
            return (1/12)*(m[i-2,1] -8*m[i-1,1] +8*m[i+1,1] - m[i+2,1])
        if i == 0:
            
            return (1/12)*(-25*m[i,1] + 48*m[i+1,1] - 36*m[i+2,1] + 16*m[i+3,1] - 3*m[i+4,1])
        if i == 1:
           
            return (1/12)*(-3*m[i-1,1] - 10*m[i,1] + 18*m[i+1,1] - 6*m[i+3,1] + m[i+4,1])
        if i == m.shape[0]-2:
            
            return (1/12)*(-m[i-3,1] + 6*m[i-2,1] - 18*m[i-1,1] + 10*m[i,1] + 3*m[i+1,1])
        if i == m.shape[0]-1:
            
            return (1/12)*(3*m[i-4,1] - 16*m[i-3,1] + 36*m[i-2,1] - 48*m[i-1,1] + 25*m[i,1])

@cuda.jit(device=True)
def partialderivative2D(i,j,mcomponent, m, k):
    if k == 0:
        if i >= 2 or i <= m.shape[0]-3:
            return (1/12)*(m[i-2,j,mcomponent] -8*m[i-1,j,mcomponent] +8*m[i+1,j,mcomponent] - m[i+2,j,mcomponent])
        if i == 0:
            return (1/12)*(-25*m[i,j,mcomponent] + 48*m[i+1,j,mcomponent] - 36*m[i+2,j,mcomponent] + 16*m[i+3,j,mcomponent] - 3*m[i+4,j,mcomponent])       
        if i == 1:
            return (1/12)*(-3*m[i-1,j,mcomponent] - 10*m[i,j,mcomponent] + 18*m[i+1,j,mcomponent] - 6*m[i+3,j,mcomponent] + m[i+4,j,mcomponent])
        if i == m.shape[0]-2:
            return (1/12)*(-m[i-3,j,mcomponent] + 6*m[i-2,j,mcomponent] - 18*m[i-1,j,mcomponent] + 10*m[i,j,mcomponent] + 3*m[i+1,j,mcomponent])
        if i == m.shape[0]-1:
            return (1/12)*(3*m[i-4,j,mcomponent] - 16*m[i-3,j,mcomponent] + 36*m[i-2,j,mcomponent] - 48*m[i-1,j,mcomponent] + 25*m[i,j,mcomponent])
    elif k == 1:
        if j >= 2 or j <= m.shape[1]-3:
            return (1/12)*(m[i,j-2,mcomponent] -8*m[i,j-1,mcomponent] +8*m[i,j+1,mcomponent] - m[i,j+2,mcomponent])
        if j == 0:
            return (1/12)*(-25*m[i,j,mcomponent] + 48*m[i,j+1,mcomponent] - 36*m[i,j+2,mcomponent] + 16*m[i,j+3,mcomponent] - 3*m[i,j+4,mcomponent])
        if j == 1:
            return (1/12)*(-3*m[i,j-1,mcomponent] - 10*m[i,j,mcomponent] + 18*m[i,j+1,mcomponent] - 6*m[i,j+3,mcomponent] + m[i,j+4,mcomponent])
        if j == m.shape[1]-2:
            return (1/12)*(-m[i,j-3,mcomponent] + 6*m[i,j-2,mcomponent] - 18*m[i,j-1,mcomponent] + 10*m[i,j,mcomponent] + 3*m[i,j+1,mcomponent])
        if j == m.shape[1]-1:
            return (1/12)*(3*m[i,j-4,mcomponent] - 16*m[i,j-3,mcomponent] + 36*m[i,j-2,mcomponent] - 48*m[i,j-1,mcomponent] + 25*m[i,j,mcomponent])
    else:
        return 0

@cuda.jit(device = True)
def cudaSecondDMI2D(m,dim):
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    j = cuda.threadIdx.y + cuda.blockDim.y*cuda.blockIdx.y
    if dim == 0:
        return partialderivative2D(i,j,2, m, 1) #partial_y Mz
    elif dim == 1:
        return -1*partialderivative2D(i,j,2, m, 0) #-partial_x Mz
    else:
        return partialderivative2D(i,j,1, m, 0) - partialderivative2D(i,j,0, m, 1) #partial_x My - partial_y Mx

@cuda.jit(device = True)
def secondDMI1DCuda(m,result,border_size):
    """Performs the cross-product-part of the DMI calculations using dx = dy = dz = 1 due to the non-dimensionality of the equations
    This is currently written using the central stencil using a point to the left and right of the point in question.

    Args:
        m ([np.ndarray(Nx,Ny,Nz,3)]): [The entire grid of spins at time i]
        border_size ([int], optional): [The size of the borders where the spins are fixed]. Defaults to border_size.

    Returns:
        [type]: [description]
    """
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    const = 1/(12)
    N = m.shape[0]
    #y-direction
    if i >= 2 and i <= N-3:
        result[i,1] = (const)*(m[i-2,2] -8*m[i-1,2] +8*m[i+1,2] - m[i+2,2])
        result[i,2] = (const)*(m[i-2,1] -8*m[i-1,1] +8*m[i+1,1] - m[i+2,1])
    if i == 0:
        result[i,1] = (const)*(-25*m[i,2] + 48*m[i+1,2] - 36*m[i+2,2] + 16*m[i+3,2] - 3*m[i+4,2])
        result[i,2] = (const)*(-25*m[i,1] + 48*m[i+1,1] - 36*m[i+2,1] + 16*m[i+3,1] - 3*m[i+4,1])
    if i == 1:
        result[i,1] = (const)*(-3*m[i-1,2] - 10*m[i,2] + 18*m[i+1,2] - 6*m[i+2,2] + m[i+3,2])
        result[i,2] = (const)*(-3*m[i-1,1] - 10*m[i,1] + 18*m[i+1,1] - 6*m[i+2,1] + m[i+3,1])

    if i == N-2:
        result[i,1] = (const)*(-m[i-3,2] + 6*m[i-2,2] - 18*m[i-1,2] + 10*m[i,2] + 3*m[i+1,2])
        result[i,2] = (const)*(-m[i-3,1] + 6*m[i-2,1] - 18*m[i-1,1] + 10*m[i,1] + 3*m[i+1,1])
    
    if i == N-1:
        result[i,1] = (const)*(3*m[i-4,2] - 16*m[i-3,2] + 36*m[i-2,2] - 48*m[i-1,2] + 25*m[i,2])
        result[i,2] = (const)*(3*m[i-4,1] - 16*m[i-3,1] + 36*m[i-2,1] - 48*m[i-1,1] + 25*m[i,1])

@cuda.jit(device = True)
def cudaLaplace2D(m,geom,dim):
    i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    j = cuda.threadIdx.y + cuda.blockDim.y*cuda.blockIdx.y
    result = 0
    #i-derivatives first
    """if (i >= 2 and i < m.shape[0]-3):
        result += 4*m[i,j,dim] - m[i-2,j,dim]-m[i-1,j,dim]-m[i+1,j,dim]-m[i+2,j,dim]
    elif (i == 1):
        result += 4*m[i,j,dim] - m[i-1,j,dim] - m[i+1,j,dim] - m[i+2,j,dim] - m[i+3,j,dim]
    elif(i==0):
        result += 4*m[i,j,dim] - m[i+1,j,dim] - m[i+2 ,j,dim] - m[i+3,j,dim] - m[i+4,j,dim]
    elif(i == m.shape[0]-2):
        result += 4*m[i,j,dim] - m[i+1,j,dim] - m[i-1,j,dim] - m[i-2,j,dim] - m[i-3,j,dim]
    elif(i == m.shape[0]-1):
        result += 4*m[i,j,dim] - m[i-1,j,dim] - m[i-2 ,j,dim] - m[i-3,j,dim] - m[i-4,j,dim]

    if (j >= 2 and j < m.shape[1]-3):
        result += 4*m[i,j,dim] - m[i,j-2,dim]-m[i,j-1,dim]-m[i,j+1,dim]-m[i,j+2,dim]
    elif (j == 1):
        result += 4*m[i,j,dim] - m[i,j-1,dim] - m[i,j+1,dim] - m[i,j+2,dim] - m[i,j+3,dim]
    elif(j==0):
        result += 4*m[i,j,dim] - m[i,j+1,dim] - m[i ,j+2,dim] - m[i,j+3,dim] - m[i,j+4,dim]
    elif(j == m.shape[1]-2):
        result += 4*m[i,j,dim] - m[i,j+1,dim] - m[i,j-1,dim] - m[i,j-2,dim] - m[i,j-3,dim]
    elif(j == m.shape[1]-1):
        result += 4*m[i,j,dim] - m[i,j-1,dim] - m[i ,j-2,dim] - m[i,j-3,dim] - m[i,j-4,dim]"""
    if i>= 0 and i < m.shape[0] and j>= 0 and j<m.shape[1]:
    #First deal with edge cases, with a return
        if i == 0 and j ==0:
            return m[i,j+1,dim] + m[i+1,j,dim] - 2*m[i,j,dim]
        if i ==0 and j == m.shape[1] -1:
            return m[i,j-1,dim] + m[i+1,j,dim] - 2*m[i,j,dim]
        if i == m.shape[0] - 1 and j ==0:
            return m[i-1,j,dim] + m[i,j+1,dim] - 2*m[i,j,dim]
        if i == m.shape[0] - 1 and j == m.shape[1]- 1:
            return m[i-1,j,dim] + m[i,j-1,dim] - 2*m[i,j,dim]
        #Now that corners are out, on to edges
        if i == 0:
            return m[i,j-1,dim] + m[i,j+1,dim] + m[i+1,j,dim] - 3*m[i,j,dim]
        if i == m.shape[0] -1:
            return m[i,j-1,dim] + m[i,j+1,dim] + m[i-1,j,dim] - 3*m[i,j,dim]
        if j == 0:
            return m[i-1,j,dim] + m[i+1,j,dim] + m[i,j+1,dim] - 3*m[i,j,dim]
        if j == m.shape[1] -1:
            return m[i+1,j,dim] + m[i-1,j,dim] + m[i,j-1,dim] -3*m[i,j,dim]
        #Then bulk here:
        coef = 4
        if geom[i+1,j,0] == 0:
            coef -=1
        if geom[i-1,j,0] ==0:
            coef -= 1
        if geom[i,j+1,0] ==0:
            coef -= 1
        if geom[i,j-1,0] == 0:
            coef -= 1
        return m[i+1,j,dim] + m[i-1,j,dim] + m[i,j+1,dim] + m[i,j-1,dim] - coef*m[i,j,dim]

    return result

@cuda.jit(device = True)
def cudaLaplace2Dold(m,dim):
    i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    j = cuda.threadIdx.y + cuda.blockDim.y*cuda.blockIdx.y
    result = 0
    #i-derivatives first
    if (i >= 2 and i < m.shape[0]-3):
        result += 4*m[i,j,dim] - m[i-2,j,dim]-m[i-1,j,dim]-m[i+1,j,dim]-m[i+2,j,dim]
    elif (i == 1):
        result += 4*m[i,j,dim] - m[i-1,j,dim] - m[i+1,j,dim] - m[i+2,j,dim] - m[i+3,j,dim]
    elif(i==0):
        result += 4*m[i,j,dim] - m[i+1,j,dim] - m[i+2 ,j,dim] - m[i+3,j,dim] - m[i+4,j,dim]
    elif(i == m.shape[0]-2):
        result += 4*m[i,j,dim] - m[i+1,j,dim] - m[i-1,j,dim] - m[i-2,j,dim] - m[i-3,j,dim]
    elif(i == m.shape[0]-1):
        result += 4*m[i,j,dim] - m[i-1,j,dim] - m[i-2 ,j,dim] - m[i-3,j,dim] - m[i-4,j,dim]

    if (j >= 2 and j < m.shape[1]-3):
        result += 4*m[i,j,dim] - m[i,j-2,dim]-m[i,j-1,dim]-m[i,j+1,dim]-m[i,j+2,dim]
    elif (j == 1):
        result += 4*m[i,j,dim] - m[i,j-1,dim] - m[i,j+1,dim] - m[i,j+2,dim] - m[i,j+3,dim]
    elif(j==0):
        result += 4*m[i,j,dim] - m[i,j+1,dim] - m[i ,j+2,dim] - m[i,j+3,dim] - m[i,j+4,dim]
    elif(j == m.shape[1]-2):
        result += 4*m[i,j,dim] - m[i,j+1,dim] - m[i,j-1,dim] - m[i,j-2,dim] - m[i,j-3,dim]
    elif(j == m.shape[1]-1):
        result += 4*m[i,j,dim] - m[i,j-1,dim] - m[i ,j-2,dim] - m[i,j-3,dim] - m[i,j-4,dim]
    
    return result

@cuda.jit(device = True)
def cudaExchange2D(m_1,m_2,geom,w_ex, A_1, A_2,dim):
    i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    j = cuda.threadIdx.y + cuda.blockDim.y*cuda.blockIdx.y
    return -(w_ex*m_2[i,j,dim] -A_1*cudaLaplace2D(m_1, geom, dim) - A_2*cudaLaplace2D(m_2, geom, dim))

@cuda.jit(device= True)
def cudaZeemanEnergy2D(h,i,j,t,dim):
    """Returns the h-function at the current time that is going to be used for thread i,j.

    Args:
        h ([Nx,3]): [description]
        i ([int]): [Thread index i]
        j ([int]): [Thread index j]
        t ([int]): [The current time that we are in]
    """
    if dim == 0:
        return h[i,j,t,0]
    elif dim ==1:
        return h[i,j,t,1]
    else:
        return h[i,j,t,2]

@cuda.jit(device =True)
def cudaf(m_1, m_2,timestep,is_a,res, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D, T, rand):
    """Calculates mdot, ie f(m_1,m_2,i) which is used in the RK4 routine. 

    Args:
        m_1 (np.ndarray): a time-instance of the m_a spins-matrix of size (N,3)
        m_2 ([type]): a time-instance of the m_b spins-matrix of size (N,3)
        i ([type]): a time-instance of the simulation

    Returns:
        [type]: Returns mdot, a (N,3) ndarray 
                Returns the current at this time step given my m cross m_dot
    """
    #Load constants 
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    res[i,0] = cudaExchange2(m_1, m_2, w_ex, A_1, A_2, border_size, 0) + cudaZeeman(i,h,border_size, 0) + cudaAnisotropy1D(m_1,m_2,w_1,w_2,0) + cudaDMIEnergy1D(m_2,is_a,d,0) + D*CudaSecondDMI1D(m_1,0) + T[i,0]*rd.xoroshiro128p_normal_float64(rand, i)
    res[i,1] = cudaExchange2(m_1, m_2, w_ex, A_1, A_2, border_size, 1) + cudaZeeman(i,h, border_size,1) + cudaAnisotropy1D(m_1,m_2,w_1,w_2,1) + cudaDMIEnergy1D(m_2,is_a,d,1) + D*CudaSecondDMI1D(m_1,1) + T[i,1]*rd.xoroshiro128p_normal_float64(rand, i)
    res[i,2] = cudaExchange2(m_1, m_2, w_ex, A_1, A_2, border_size, 2) + cudaZeeman(i,h, border_size,2) + cudaAnisotropy1D(m_1,m_2,w_1,w_2,2) + cudaDMIEnergy1D(m_2,is_a,d,2) + D*CudaSecondDMI1D(m_1,2) + T[i,2]*rd.xoroshiro128p_normal_float64(rand, i)
    
    
    cuda.syncthreads()
    if i >= 0 and i <m_1.shape[0]:
        K_x = 1/(1+alpha[i,0]**2)
        K_y = 1/(1+alpha[i,1]**2)
        K_z = 1/(1+alpha[i,2]**2)
        beta_x = beta/K_x
        beta_y = beta/K_y
        beta_z = beta/K_z
        #H_1 + C*J[:,i,:] + alpha[:,:]*(crossProduct(m_1, H_1 +beta*J[:,i,:]))
        temp_x = res[i,0] + C*J[i,timestep,0]+ alpha[i,0]*((m_1[i,1]*(res[i,2] + beta_x*J[i,timestep,2]) - m_1[i,2]*(res[i,1] + beta_x*J[i,timestep,1])))
        temp_y = res[i,1] + C*J[i,timestep,1]+ alpha[i,1]*((m_1[i,2]*(res[i,0] + beta_x*J[i,timestep,0]) - m_1[i,0]*(res[i,2] + beta_x*J[i,timestep,2])))
        temp_z = res[i,2] + C*J[i,timestep,2]+ alpha[i,2]*((m_1[i,0]*(res[i,1] + beta_x*J[i,timestep,1]) - m_1[i,1]*(res[i,0] + beta_x*J[i,timestep,0])))
        res[i,0] = -(K_x)*(m_1[i,1]*temp_z - temp_y*m_1[i,2])
        res[i,1] = (K_y)*(m_1[i,0]*temp_z - temp_x*m_1[i,2])
        res[i,2] = -(K_z)*(m_1[i,0]*temp_y - temp_x*m_1[i,1])

    return res

@cuda.jit(device =True)
def cudaf2D(m_1, m_2,geom,timestep,is_a,res, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D,Temp,rand):
    """Calculates mdot, ie f(m_1,m_2,i) which is used in the RK4 routine. 

    Args:
        m_1 (np.ndarray): a time-instance of the m_a spins-matrix of size (N,3)
        m_2 ([type]): a time-instance of the m_b spins-matrix of size (N,3)
        i ([type]): a time-instance of the simulation

    Returns:
        [type]: Returns mdot, a (N,3) ndarray 
                Returns the current at this time step given my m cross m_dot
    """
    #Load constants 
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    j =  cuda.blockDim.y*cuda.blockIdx.y + cuda.threadIdx.y
    res[i,j,0] = cudaExchange2D(m_1,m_2,geom,w_ex,A_1,A_2,0) + cudaZeemanEnergy2D(h,i,j,timestep,0) + cudaAnisotropy2D(m_1, m_2, w_1, w_2, 0) + Temp[i,j,0]*rd.xoroshiro128p_normal_float64(rand, i+j*m_1.shape[0]) + cudaDMIEnergy2D(m_1, is_a, d,0) + D*cudaSecondDMI2D(m_1,0)
    res[i,j,1] = cudaExchange2D(m_1,m_2,geom,w_ex,A_1,A_2,1) + cudaZeemanEnergy2D(h,i,j,timestep,1) + cudaAnisotropy2D(m_1, m_2, w_1, w_2, 1) + Temp[i,j,1]*rd.xoroshiro128p_normal_float64(rand, i+j*m_1.shape[0]) + cudaDMIEnergy2D(m_1, is_a, d,1) + D*cudaSecondDMI2D(m_1,1)
    res[i,j,2] = cudaExchange2D(m_1,m_2,geom,w_ex,A_1,A_2,2) + cudaZeemanEnergy2D(h,i,j,timestep,2) + cudaAnisotropy2D(m_1, m_2, w_1, w_2, 2) + Temp[i,j,2]*rd.xoroshiro128p_normal_float64(rand, i+j*m_1.shape[0]) + cudaDMIEnergy2D(m_1, is_a, d,2) + D*cudaSecondDMI2D(m_1,2)
    
    
    cuda.syncthreads()
    if i >= 0 and i <m_1.shape[0] and j>=0 and j<m_1.shape[1]:
        K_x = 1/(1+alpha[i,j,0]**2)
        K_y = 1/(1+alpha[i,j,1]**2)
        K_z = 1/(1+alpha[i,j,2]**2)
        beta_x = beta/K_x
        beta_y = beta/K_y
        beta_z = beta/K_z
        #H_1 + C*J[:,i,:] + alpha[:,:]*(crossProduct(m_1, H_1 +beta*J[:,i,:]))
        temp_x = res[i,j,0] + C*J[i,j,timestep,0]+ alpha[i,j,0]*((m_1[i,j,1]*(res[i,j,2] + beta_x*J[i,j,timestep,2]) - m_1[i,j,2]*(res[i,j,1] + beta_x*J[i,j,timestep,1])))
        temp_y = res[i,j,1] + C*J[i,j,timestep,1]+ alpha[i,j,1]*((m_1[i,j,2]*(res[i,j,0] + beta_x*J[i,j,timestep,0]) - m_1[i,j,0]*(res[i,j,2] + beta_x*J[i,j,timestep,2])))
        temp_z = res[i,j,2] + C*J[i,j,timestep,2]+ alpha[i,j,2]*((m_1[i,j,0]*(res[i,j,1] + beta_x*J[i,j,timestep,1]) - m_1[i,j,1]*(res[i,j,0] + beta_x*J[i,j,timestep,0])))
        res[i,j,0] = -(K_x)*(m_1[i,j,1]*temp_z - temp_y*m_1[i,j,2])
        res[i,j,1] = (K_y)*(m_1[i,j,0]*temp_z - temp_x*m_1[i,j,2])
        res[i,j,2] = -(K_z)*(m_1[i,j,0]*temp_y - temp_x*m_1[i,j,1])
    return res


@cuda.jit(fastmath = True)    
def cudaTimeStep(m_1, m_2,k1,k2,k3,k4,dt,i, is_a, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D,T,rand,res, resdot):
    """A timestep of the RK45 routine. Uses f(m_1,m_2,i) to calculate a timestep

    Args:
        m_1 (np.ndarray(N,3)): m_1 is either m_a or m_b, depending on wether this function is called to calculate m_a dot or m_b dot respectively
        m_2 (np.ndarray(N,3)): the other spin-sublattice
        dt (float): a fixed timetep, used in RK45
        i (int): The i'th time

    Returns:
        [type]: [description]
    """
    """threadsperblock = 32
    blockspergrid = (m_1_in.size +(threadsperblock -1))// threadsperblock
    m_1= np.ascontiguousarray(m_1_in)
    m_2 = np.ascontiguousarray(m_2_in)
    k1 = np.zeros_like(m_1)
    k2 = np.zeros_like(m_1)
    k3 = np.zeros_like(m_1)
    k4 = np.zeros_like(m_1)"""
    j = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    N_x = m_1.shape[0]
    if j >= 0:
        k1 = cudaf(m_1[:], m_2[:],i,is_a,k1, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D,T, rand)
        res[j,0] = m_1[j,0] + k1[j,0]*0.5*dt
        res[j,1] = m_1[j,1] + k1[j,1]*0.5*dt
        res[j,2] = m_1[j,2] + k1[j,2]*0.5*dt
    
  
        k2 = cudaf(res, m_2[:], i,is_a,k2, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D,T,rand)
        res[j,0] = m_1[j,0] + k2[j,0]*0.5*dt
        res[j,1] = m_1[j,1] + k2[j,1]*0.5*dt
        res[j,2] = m_1[j,2] + k2[j,2]*0.5*dt
    
    
        k3 = cudaf(res, m_2[:], i,is_a,k3, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D,T,rand)
    
        res[j,0] = m_1[j,0] + k3[j,0]*dt
        res[j,1] = m_1[j,1] + k3[j,1]*dt
        res[j,2] = m_1[j,2] + k3[j,2]*dt
    
    
        k4 = cudaf(res, m_2[:], i,is_a,k4, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D,T,rand)
    
    
    #m_1[:] += (dt / 6) * (k1 + 2* k2 + 2 * k3 + k4)
    
        res[j,0] = m_1[j,0] + (dt / 6) * (k1[j,0] + 2* k2[j,0] + 2 * k3[j,0] + k4[j,0])
        res[j,1] = m_1[j,1] + (dt / 6) * (k1[j,1] + 2* k2[j,1] + 2 * k3[j,1] + k4[j,1])
        res[j,2] = m_1[j,2] + (dt / 6) * (k1[j,2] + 2* k2[j,2] + 2 * k3[j,2] + k4[j,2])
        
        #normalize
        abs_val = sqrt(res[j,0]**2 + res[j,1]**2 + res[j,2]**2)
        
        res[j,0] = res[j,0]/abs_val
        res[j,1] = res[j,1]/abs_val
        res[j,2] = res[j,2]/abs_val

        resdot[j,0] = k1[j,0]
        resdot[j,1] = k1[j,1]
        resdot[j,2] = k1[j,2]

@cuda.jit(fastmath = True)    
def cudaTimeStep2D(m_1, m_2,geom,k1,k2,k3,k4,dt,t_i, is_a, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D,Temp,rand, res, resdot):
    """A timestep of the RK45 routine. Uses f(m_1,m_2,i) to calculate a timestep

    Args:
        m_1 (np.ndarray(N,3)): m_1 is either m_a or m_b, depending on wether this function is called to calculate m_a dot or m_b dot respectively
        m_2 (np.ndarray(N,3)): the other spin-sublattice
        dt (float): a fixed timetep, used in RK45
        i (int): The i'th time

    Returns:
        [type]: [description]
    """
    """threadsperblock = 32
    blockspergrid = (m_1_in.size +(threadsperblock -1))// threadsperblock
    m_1= np.ascontiguousarray(m_1_in)
    m_2 = np.ascontiguousarray(m_2_in)
    k1 = np.zeros_like(m_1)
    k2 = np.zeros_like(m_1)
    k3 = np.zeros_like(m_1)
    k4 = np.zeros_like(m_1)"""
    i = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
    j = cuda.blockDim.y*cuda.blockIdx.y + cuda.threadIdx.y
    N_x = m_1.shape[0]
    N_y = m_1.shape[1]
    if (j >= 0) and (j < N_y) and (i>=0) and (i < N_x):
        k1 = cudaf2D(m_1[:,:], m_2[:,:],geom,t_i,is_a,k1, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D,Temp, rand)
        res[i,j,0] = m_1[i,j,0] + k1[i,j,0]*0.5*dt
        res[i,j,1] = m_1[i,j,1] + k1[i,j,1]*0.5*dt
        res[i,j,2] = m_1[i,j,2] + k1[i,j,2]*0.5*dt
    
  
        k2 = cudaf2D(res, m_2[:],geom, t_i,is_a,k2, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D,Temp,rand)
        res[i,j,0] = m_1[i,j,0] + k2[i,j,0]*0.5*dt
        res[i,j,1] = m_1[i,j,1] + k2[i,j,1]*0.5*dt
        res[i,j,2] = m_1[i,j,2] + k2[i,j,2]*0.5*dt
    
    
        k3 = cudaf2D(res, m_2[:,:],geom, t_i,is_a,k3, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D,Temp,rand)
    
        res[i,j,0] = m_1[i,j,0] + k3[i,j,0]*dt
        res[i,j,1] = m_1[i,j,1] + k3[i,j,1]*dt
        res[i,j,2] = m_1[i,j,2] + k3[i,j,2]*dt
    
    
        k4 = cudaf2D(res, m_2[:,:],geom, t_i,is_a,k4, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D,Temp,rand)
    
    
    #m_1[:] += (dt / 6) * (k1 + 2* k2 + 2 * k3 + k4)
    
        res[i,j,0] = m_1[i,j,0] + (dt / 6) * (k1[i,j,0] + 2* k2[i,j,0] + 2 * k3[i,j,0] + k4[i,j,0])
        res[i,j,1] = m_1[i,j,1] + (dt / 6) * (k1[i,j,1] + 2* k2[i,j,1] + 2 * k3[i,j,1] + k4[i,j,1])
        res[i,j,2] = m_1[i,j,2] + (dt / 6) * (k1[i,j,2] + 2* k2[i,j,2] + 2 * k3[i,j,2] + k4[i,j,2])

        #normalize
        abs_val = sqrt(res[i,j,0]**2 + res[i,j,1]**2 + res[i,j,2]**2)
        
        if geom[i,j,0] != 0:
            res[i,j,0] = res[i,j,0]/abs_val
            res[i,j,1] = res[i,j,1]/abs_val
            res[i,j,2] = res[i,j,2]/abs_val

        resdot[i,j,0] = k1[i,j,0]
        resdot[i,j,1] = k1[i,j,1]
        resdot[i,j,2] = k1[i,j,2]

def cudaTimeEvolution(m_a, m_b, N_steps, dt, cut_borders_at = 0, border_size = 0,stride = 0, constants = ()):
    """Calculates the state of both sublattice spins m_a, m_b for all timesteps N_steps using the RK45 routine.

    Args:
        m_a (np.ndarray(N,N_steps,3)): the m_a sublattice spins matrix of size (N,N_steps,3)
        m_b (np.ndarray(N,N_steps,3)): the m_b sublattice spins matrix of size (N,N_steps,3)
        N_steps (int): The number of iterations to perform using a fixed timestep
        dt (float): One time-step increment

    Returns:
        m_a (np.ndarray(N,N_steps,3)): the m_a sublattice spins matrix of size (N,N_steps,3), now with the spin-matrices updated for all times
        m_b (np.ndarray(N,N_steps,3)): the m_b sublattice spins matrix of size (N,N_steps,3), now with the spin-matrices updated for all times
        T (np.ndarray(N_steps)): This is a numpy array with all the times of the simulation
    """
    T = np.zeros(N_steps)
    mdot_a = np.zeros((np.shape(m_a)))
    mdot_b = np.zeros((np.shape(m_b)))
    
    
    #Initialising kernel values TODO: Opmtimalization by using max occupancy
    threadsperblock = 32
    blockspergrid = (m_a[:,0,0].size +(threadsperblock -1))// threadsperblock
    j = 0
    
    #Initialising loop arrays
    temp_m_a = np.zeros((m_a.shape[0], 3))
    temp_m_b = np.zeros_like(temp_m_a)
    temp_m_a_dot = np.zeros_like(temp_m_a)
    temp_m_b_dot = np.zeros_like(temp_m_a)
    inter_m_a = np.zeros_like(temp_m_a)
    temp_m_a = np.ascontiguousarray(m_a[:,0,:])
    temp_m_b = np.ascontiguousarray(m_b[:,0,:])
    res_a,resdot_a = np.zeros((m_a.shape[0],3)),np.zeros((m_b.shape[0],3))
    res_a = np.ascontiguousarray(res_a)
    resdot_a = np.ascontiguousarray(resdot_a)
    res_b = res_a.copy()
    resdot_b = res_a.copy()

    #Initialising device arrays TODO: Shared memory when numba cuda allows for it
    k1 = np.ascontiguousarray(np.zeros_like(res_a))
    k2 = np.ascontiguousarray(np.zeros_like(res_a))
    k3 = np.ascontiguousarray(np.zeros_like(res_a))
    k4 = np.ascontiguousarray(np.zeros_like(res_a))
    d_k1 = cuda.to_device(k1)
    d_k2 = cuda.to_device(k2)
    d_k3 = cuda.to_device(k3)
    d_k4 = cuda.to_device(k4)
    w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D, Temp = constants
    d_J = cuda.to_device(J)
    d_h = cuda.to_device(h)
    d_T = cuda.to_device(Temp)
    d_alpha = cuda.to_device(alpha)
    rand_seed = random.randrange(sys.maxsize)
    d_rand = rd.create_xoroshiro128p_states(threadsperblock*blockspergrid,rand_seed)
    for i in tqdm(range(N_steps)):
        if i == cut_borders_at:
            border_size = 0
        inter_m_a = temp_m_a    
        
        cudaTimeStep[blockspergrid,threadsperblock](temp_m_a, temp_m_b,d_k1,d_k2,d_k3,d_k4, dt, i, True, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,d_alpha,d_J,d,d_h,D,d_T,d_rand, res_a, resdot_a)
        temp_m_a, temp_m_a_dot= res_a,resdot_a
        
        cudaTimeStep[blockspergrid,threadsperblock](temp_m_b, inter_m_a,d_k1,d_k2,d_k3,d_k4, dt, i, False,border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,d_alpha,d_J,d,d_h,D,d_T,d_rand, res_b, resdot_b)
        temp_m_b, temp_m_b_dot= res_b,resdot_b
    
        if i % stride == 0:
            T[j] += (i)*dt
            m_a[:,j,:] = temp_m_a
            mdot_a[:,j,:] = temp_m_a_dot
            m_b[:,j,:] = temp_m_b
            mdot_b[:,j,:] = temp_m_b_dot
            j += 1
    return m_a, m_b, T,mdot_a,mdot_b

def cudaTimeEvolution2D(m_a,m_b, N_steps, dt, geom, cut_borders_at=0, border_size = 0, stride = 0, constants = ()):
    T = np.zeros(N_steps)
    mdot_a = np.zeros((np.shape(m_a)))
    mdot_b = np.zeros((np.shape(m_b)))
    
    
    #Initialising kernel values
    threadsperblock = 8
    N_x = m_a.shape[0]
    N_y = m_a.shape[1]
    blockspergrid_x = (N_x +(threadsperblock -1))// threadsperblock
    blockspergrid_y = (N_y +(threadsperblock -1))// threadsperblock
    blockspergrid = blockspergrid_x,blockspergrid_y
    threadblock = threadsperblock, threadsperblock
    j = 0
    #Initialising loop arrays
    temp_m_a = np.zeros((m_a.shape[0], 3))
    temp_m_b = np.zeros_like(temp_m_a)
    temp_m_a_dot = np.zeros_like(temp_m_a)
    temp_m_b_dot = np.zeros_like(temp_m_a)
    inter_m_a = np.zeros_like(temp_m_a)
    temp_m_a = np.ascontiguousarray(m_a[:,:,0,:])
    temp_m_b = np.ascontiguousarray(m_b[:,:,0,:])
    res_a,resdot_a = np.zeros((m_a.shape[0],m_a.shape[1],3)),np.zeros((m_b.shape[0],m_a.shape[1],3))
    res_a = np.ascontiguousarray(res_a)
    resdot_a = np.ascontiguousarray(resdot_a)
    res_b = res_a.copy()
    resdot_b = res_a.copy()

    #Initialising device arrays
    k1 = np.ascontiguousarray(np.zeros_like(res_a))
    k2 = np.ascontiguousarray(np.zeros_like(res_a))
    k3 = np.ascontiguousarray(np.zeros_like(res_a))
    k4 = np.ascontiguousarray(np.zeros_like(res_a))
    d_k1 = cuda.to_device(k1)
    d_k2 = cuda.to_device(k2)
    d_k3 = cuda.to_device(k3)
    d_k4 = cuda.to_device(k4)
    w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D, Temp = constants
    d_geom = np.ascontiguousarray(geom)
    d_geom = cuda.to_device(d_geom)
    d_J = cuda.to_device(J)
    d_h = cuda.to_device(h)
    d_alpha = cuda.to_device(alpha)
    d_T = cuda.to_device(Temp)
    rand_seed = random.randrange(sys.maxsize)
    d_rand = rd.create_xoroshiro128p_states(blockspergrid_x*16*16*blockspergrid_y, rand_seed)
    for i in tqdm(range(N_steps)):
        if i == cut_borders_at:
            border_size = 0
        inter_m_a = temp_m_a    
        
        cudaTimeStep2D[blockspergrid,threadblock](temp_m_a, temp_m_b,d_geom,d_k1,d_k2,d_k3,d_k4, dt, i, True, border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,d_alpha,d_J,d,d_h,D,d_T,d_rand, res_a, resdot_a)
        temp_m_a, temp_m_a_dot= res_a,resdot_a
        
        cudaTimeStep2D[blockspergrid,threadblock](temp_m_b, inter_m_a,d_geom,d_k1,d_k2,d_k3,d_k4, dt, i, False,border_size, w_ex,A_1,A_2,w_1,w_2,beta,C,d_alpha,d_J,d,d_h,D,d_T,d_rand, res_b, resdot_b)
        temp_m_b, temp_m_b_dot= res_b,resdot_b
    
        if i % stride == 0:
            T[j] += (i)*dt
            m_a[:,:,j,:] = temp_m_a
            mdot_a[:,:,j,:] = temp_m_a_dot
            m_b[:,:,j,:] = temp_m_b
            mdot_b[:,:,j,:] = temp_m_b_dot
            j += 1
    return m_a, m_b, T,mdot_a,mdot_b

def cudaSolver1D(N_steps, dt, init_state,  cut_borders_at = 0, border_size = 0, stride = 0, constants = ()):
    """A wrapper function that is needed to call to both initialize the system as well as solve it. 

    Args:
        N (int): Number of lattice sites /spins
        N_steps (int): The number of time-steps used in the RK45 routine
        dt (float): the timestep used in RK45
        initialiser (function): A potentially custom function that is used to initalize the spins. 
        initargs (tuple, optional): [description]. Defaults to (). A tuple of the necessary arguments for the initializer function. 

    Returns:
        m_a (np.ndarray(N,N_steps,3)): All the spins after the simulation on the a-sublattice
        m_b (np.ndarray(N,N_steps,3)): All the spins after the simulation on the b-sublattice
        T (np.ndarray(N_steps)): A vector of all the times in the simulation
    """
    m_a_start,m_b_start = init_state[0], init_state[1]    
    m_a, m_b, T,mdot_a,mdot_b = cudaTimeEvolution(m_a_start, m_b_start, N_steps, dt, cut_borders_at = cut_borders_at, border_size = border_size,stride = stride, constants = constants)
    return m_a, m_b, T, mdot_a, mdot_b

def cudaSolver2D(N_steps, dt, init_state, geom, cut_borders_at = 0, border_size = 0, stride = 0, constants = ()):
    """A wrapper function that is needed to call to both initialize the system as well as solve it. 

    Args:
        N (int): Number of lattice sites /spins
        N_steps (int): The number of time-steps used in the RK45 routine
        dt (float): the timestep used in RK45
        initialiser (function): A potentially custom function that is used to initalize the spins. 
        initargs (tuple, optional): [description]. Defaults to (). A tuple of the necessary arguments for the initializer function. 

    Returns:
        m_a (np.ndarray(N,Ny,N_steps/stride,3)): All the spins after the simulation on the a-sublattice
        m_b (np.ndarray(N,Ny,N_steps/stride,3)): All the spins after the simulation on the b-sublattice
        T (np.ndarray(N_steps)): A vector of all the times in the simulation
    """
    m_a_start, m_b_start = init_state[0], init_state[1]
    m_a, m_b, T, mdot_a, mdot_b = cudaTimeEvolution2D(m_a_start, m_b_start, N_steps, dt, geom, cut_borders_at=cut_borders_at, border_size=border_size,stride= stride, constants = constants)
    return m_a,m_b, T,mdot_a, mdot_b

if __name__ == '__main__':
    """t1 = time.time()
    v1 = np.ones((100,3))
    v2 = np.random.rand(100,3)
    res = np.zeros((v1.shape[0],v1.shape[1]))
    threadsperblock = 32
    blockspergrid = (v1.size +(threadsperblock -1))// threadsperblock
    #cudaCrossProduct1D[blockspergrid,threadsperblock](v1,v2,res)
    cudaLaplacian1DKernel[blockspergrid,threadsperblock](v2,res,0)
    print("Laplace:", res)
    print(time.time()-t1)  
    t3 = time.time()
    res2 = RK4.laplacian1D(v1,v2)
    print(time.time()- t3)"""


    
    print(nb.cuda.gpus)

#result = np.array(([1,2,3],[1,2,3],[1,2,3],[5,6,7],[6,8,3],[5,3,6]))
#result1 = np.zeros_like(result)
#secondDMI1DCuda[32,32](result,result1,0)
#print(result1)
"""
The answer should be this: 
array([[ 0,  5,  3],
       [ 0, -2, -1],
       [ 0,  2,  2],
       [ 0,  0,  3],
       [ 0, -5,  0],
       [ 0, 18, -9]])
"""