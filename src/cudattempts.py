import numba.cuda as cuda
import numpy as np

def CudaAnisotropy1D(i,m_a,m_b,w_1,w_2,result):
    """Updates the  anisotropy of the spin located on the lattice i

    Args:
        i ([int]): [thread idex from GPU]
        m_a ([Nx,3]): [a matrix of]
        m_b ([Nx,3]): [description]
        w_1 ([3,3]): [description]
        w_2 ([3,3]): [description]
        result ([Nx,3]): [description]
    """
    result[i,0] = -1* w_1[0,0]*m_a[i,0] - w_1[0,1]*m_a[i,1] - w_1[0,2]*m_a[i,2] - w_2[0,0]*m_b[i,0] - w_2[0,1]*m_b[i,1] - w_2[0,2]*m_b[i,2]
    result[i,1] = -1* w_1[1,0]*m_a[i,0] - w_1[1,1]*m_a[i,1] - w_1[1,2]*m_a[i,2] - w_2[1,0]*m_b[i,0] - w_2[1,1]*m_b[i,1] - w_2[1,2]*m_b[i,2]
    result[i,2] = -1* w_1[2,0]*m_a[i,0] - w_1[2,1]*m_a[i,1] - w_1[2,2]*m_a[i,2] - w_2[2,0]*m_b[i,0] - w_2[2,1]*m_b[i,1] - w_2[2,2]*m_b[i,2]

def CudaAnisotropy2D(i,j,m_a,m_b,w_1,w_2,result):
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
    result[i,j,0] = -1* w_1[0,0]*m_a[i,j,0] - w_1[0,1]*m_a[i,j,1] - w_1[0,2]*m_a[i,j,2] - w_2[0,0]*m_b[i,j,0] - w_2[0,1]*m_b[i,j,1] - w_2[0,2]*m_b[i,j,2]
    result[i,j,1] = -1* w_1[1,0]*m_a[i,j,0] - w_1[1,1]*m_a[i,j,1] - w_1[1,2]*m_a[i,j,2] - w_2[1,0]*m_b[i,j,0] - w_2[1,1]*m_b[i,j,1] - w_2[1,2]*m_b[i,j,2]
    result[i,j,2] = -1* w_1[2,0]*m_a[i,j,0] - w_1[2,1]*m_a[i,j,1] - w_1[2,2]*m_a[i,j,2] - w_2[2,0]*m_b[i,j,0] - w_2[2,1]*m_b[i,j,1] - w_2[2,2]*m_b[i,j,2]

def CudaAnisotropy3D(i,j,k,m_a,m_b,w_1,w_2,result):
    """Updates the  anisotropy of the spin located on the lattice i,j,k

    Args:
        i ([type]): [description]
        j ([type]): [description]
        k ([type]): [description]
        m_a ([type]): [description]
        m_b ([type]): [description]
        w_1 ([type]): [description]
        w_2 ([type]): [description]
        result ([type]): [description]
    """
    result[i,j,k,0] = -1* w_1[0,0]*m_a[i,j,k,0] - w_1[0,1]*m_a[i,j,k,1] - w_1[0,2]*m_a[i,j,k,2] - w_2[0,0]*m_b[i,j,k,0] - w_2[0,1]*m_b[i,j,k,1] - w_2[0,2]*m_b[i,j,k,2]
    result[i,j,k,1] = -1* w_1[1,0]*m_a[i,j,k,0] - w_1[1,1]*m_a[i,j,k,1] - w_1[1,2]*m_a[i,j,k,2] - w_2[1,0]*m_b[i,j,k,0] - w_2[1,1]*m_b[i,j,k,1] - w_2[1,2]*m_b[i,j,k,2]
    result[i,j,k,2] = -1* w_1[2,0]*m_a[i,j,k,0] - w_1[2,1]*m_a[i,j,k,1] - w_1[2,2]*m_a[i,j,k,2] - w_2[2,0]*m_b[i,j,k,0] - w_2[2,1]*m_b[i,j,k,1] - w_2[2,2]*m_b[i,j,k,2]

def CudaZeemanEnergy1D(h,i,t,result):
    """Returns the h-function at the current time that is going to be used for thread i. 

    Args:
        h ([Nx,Nt,3]): [description]
        i ([int]): [Thread index i]
        t ([int]): [The current time that we are in]
    """
    result[i,0] += h[i,t,0]
    result[i,1] += h[i,t,1]
    result[i,2] += h[i,t,2]

def CudaZeemanEnergy2D(h,i,j,t,result):
    """Returns the h-function at the current time that is going to be used for thread i,j.

    Args:
        h ([Nx,3]): [description]
        i ([int]): [Thread index i]
        j ([int]): [Thread index j]
        t ([int]): [The current time that we are in]
    """
    result[i,j,0] += h[i,j,t,0]
    result[i,j,1] += h[i,j,t,1]
    result[i,j,2] += h[i,j,t,2]

def CudaZeemanEnergy3D(h,i,j,k,t,result):
    """Returns the h-function at the current time that is going to be used for thread i,j,k

    Args:
        h ([Nx,Ny,Nz,Nt,3]): [description]
        i ([int]): [Thread index i]
        j ([int]): [Thread index j]
        k ([int]): [Thread index k]
        t ([int]): [The current time that we are in]
    """
    result[i,j,k,0] += h[i,j,k,t,0]
    result[i,j,k,1] += h[i,j,k,t,1]
    result[i,j,k,2] += h[i,j,k,t,2]

def CudaDMIEnergy1D(m,i,a,result):
    """Calculates the DMI term given by m x d. However, depending on m_a or m_b, 
    this term differns, and therefore we include the variable "a" to differentiate between them.

    Args:
        m ([Nx,3]): [either m_a or m_b]
        i ([int]): [thread index]
        a ([bool]): [If true, the m-matrix is m_a. if false, it is m_b]
        result ([type]): [the  return matrix. This  needs to be changed in the GPU implementation]
    """
    if a:
        result[i,0] = m[i,2]*d[1] - m[i,1]*d[2] 
        result[i,1] = m[i,0]*d[2] - m[i,2]*d[0] 
        result[i,2] = m[i,1]*d[0] - m[i,0]*d[1]
    else:
        result[i,0] = m[i,1]*d[2] - m[i,2]*d[1]
        result[i,1] = m[i,2]*d[0] - m[i,0]*d[2]
        result[i,2] = m[i,0]*d[1] - m[i,1]*d[0]

def CudaDMIEnergy2D(m,i,j,a,result):
    """Calculates the DMI term given by m x d. However, depending on m_a or m_b, 
    this term differns, and therefore we include the variable "a" to differentiate between them.

    Args:
        m ([Nx,3]): [either m_a or m_b]
        i ([int]): [thread index]
        j ([int]): [thread index]
        a ([bool]): [If true, the m-matrix is m_a. if false, it is m_b]
        result ([type]): [the  return matrix. This  needs to be changed in the GPU implementation]
    """
    if a:
        result[i,j,0] = m[i,j,2]*d[1] - m[i,j,1]*d[2] 
        result[i,j,1] = m[i,j,0]*d[2] - m[i,j,2]*d[0] 
        result[i,j,2] = m[i,j,1]*d[0] - m[i,j,0]*d[1]
    else:
        result[i,j,0] = m[i,j,1]*d[2] - m[i,j,2]*d[1]
        result[i,j,1] = m[i,j,2]*d[0] - m[i,j,0]*d[2]
        result[i,j,2] = m[i,j,0]*d[1] - m[i,j,1]*d[0]

def CudaDMIEnergy3D(m,i,j,k,a,result):
    """Calculates the DMI term given by m x d. However, depending on m_a or m_b, 
    this term differns, and therefore we include the variable "a" to differentiate between them.

    Args:
        m ([Nx,3]): [either m_a or m_b]
        i ([int]): [thread index]
        j ([int]): [thread index]
        a ([bool]): [If true, the m-matrix is m_a. if false, it is m_b]
        result ([type]): [the  return matrix. This  needs to be changed in the GPU implementation]
    """
    if a:
        result[i,j,k,0] = m[i,j,k,2]*d[1] - m[i,j,k,1]*d[2] 
        result[i,j,k,1] = m[i,j,k,0]*d[2] - m[i,j,k,2]*d[0] 
        result[i,j,k,2] = m[i,j,k,1]*d[0] - m[i,j,k,0]*d[1]
    else:
        result[i,j,k,0] = m[i,j,k,1]*d[2] - m[i,j,k,2]*d[1]
        result[i,j,k,1] = m[i,j,k,2]*d[0] - m[i,j,k,0]*d[2]
        result[i,j,k,2] = m[i,j,k,0]*d[1] - m[i,j,k,1]*d[0]

def CudaSecondDMI1D(m,i,result):
    """Performs the cross-product-part of the DMI calculations using dx = dy = dz = 1 due to the non-dimensionality of the equations
    This is currently written using the central stencil using a point to the left and right of the point in question.

    Args:
        m ([np.ndarray(Nx,3)]): [The entire grid of spins at time i]
        border_size ([int], optional): [The size of the borders where the spins are fixed]. Defaults to border_size.

    Returns:
        [type]: [description]
    """
    if i >= 2 or i <= m.shape[0]-3:
        result[i,1] = (1/12)*(m[i-2,2] -8*m[i-1,2] +8*m[i+1,2] - m[i+2,2])
        result[i,2] = (1/12)*(m[i-2,1] -8*m[i-1,1] +8*m[i+1,1] - m[i+2,1])
    if i == 0:
        result[i,1] = (1/12)*(-25*m[i,2] + 48*m[i+1,2] - 36*m[i+2,2] + 16*m[i+3,2] - 3*m[i+4,2])
        result[i,2] = (1/12)*(-25*m[i,1] + 48*m[i+1,1] - 36*m[i+2,1] + 16*m[i+3,1] - 3*m[i+4,1])
    if i == 1:
        result[i,1] = (1/12)*(-3*m[i-1,2] - 10*m[i,2] + 18*m[i+1,2] - 6*m[i+3,2] + m[i+4,2])
        result[i,2] = (1/12)*(-3*m[i-1,1] - 10*m[i,1] + 18*m[i+1,1] - 6*m[i+3,1] + m[i+4,1])
    if i == m.shape[0]-2:
        result[i,1] = (1/12)*(-m[i-3,2] + 6*m[i-2,2] - 18*m[i-1,2] + 10*m[i,2] + 3*m[i+1,2])
        result[i,2] = (1/12)*(-m[i-3,1] + 6*m[i-2,1] - 18*m[i-1,1] + 10*m[i,1] + 3*m[i+1,1])
    if i == m.shape[0]-1:
        result[i,1] = (1/12)*(3*m[i-4,2] - 16*m[i-3,2] + 36*m[i-2,2] - 48*m[i-1,2] + 25*m[i,2])
        result[i,2] = (1/12)*(3*m[i-4,1] - 16*m[i-3,1] + 36*m[i-2,1] - 48*m[i-1,1] + 25*m[i,1])

def CudaLaplace2D(m,dim):
    i, j = cuda.grid(2)
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

def CudaExchange2D(m_1,m_2,w_ex, A_1, A_2,dim):
    i,j = cuda.grid(2)
    return -(w_ex*m_2[i,j,dim] -A_1*CudaLaplace2D(m_1, dim) - A_2*CudaLaplace2D(m_2, dim))