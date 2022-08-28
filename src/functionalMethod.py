import numpy as np
import numba as nb
import time
import random
from tqdm import tqdm
import random as rd



dtype = np.float64
@nb.jit(cache = True, nopython = True,  fastmath = True)
def normalize1D(m_1):
    """Function to normalize an array of spins, this function 
    is used in the projection
    step of the Runge-Kutta 4 method used in this program.

    Args:
        m_1 (ndarray(N,3) or ndarray(Nx,Ny,3)): Timestep
        in magnetisation

    Returns:
        ndarray: Normalized input m_1
    """
    norms = np.sqrt(m_1[:,0]**2 + m_1[:,1]**2 + m_1[:,2]**2)
    m_1[:, 1] = m_1[:, 1] / norms[:]
    m_1[:, 2] = m_1[:, 2] / norms[:]
    return m_1

@nb.jit(cache = True, nopython = True,  fastmath = True)
def normalize2D(m_1,geom):
    """Function to normalize an array of spins, this function 
    is used in the projection
    step of the Runge-Kutta 4 method used in this program.

    Args:
        m_1 (ndarray(N,3) or ndarray(Nx,Ny,3)): Timestep
        in magnetisation

    Returns:
        ndarray: Normalized input m_1
    """
    
    
    #TODO: Figure out a way to avoid this nasty for loop
    for i in range(m_1.shape[0]):
        for j in range(m_1.shape[1]):
            if geom[i,j,0] != 0:
                norm = np.sqrt(m_1[i,j,0]**2 + m_1[i,j,1]**2 + m_1[i,j,2]**2)
                m_1[i,j, 0] = m_1[i,j, 0] / norm
                m_1[i,j, 1] = m_1[i,j, 1] / norm
                m_1[i,j, 2] = m_1[i,j, 2] / norm
    return m_1

@nb.jit(cache = True, nopython = True,  fastmath = True)
def normalize2Dold(m_1,geom):
    """Function to normalize an array of spins, this function 
    is used in the projection
    step of the Runge-Kutta 4 method used in this program.

    Args:
        m_1 (ndarray(N,3) or ndarray(Nx,Ny,3)): Timestep
        in magnetisation

    Returns:
        ndarray: Normalized input m_1
    """
    norms = np.sqrt(m_1[:,:,0]**2 + m_1[:,:,1]**2 + m_1[:,:,2]**2)
    m_1[:,:, 0] = m_1[:,:, 0] / norms[:,:]
    m_1[:,:, 1] = m_1[:,:, 1] / norms[:,:]
    m_1[:,:, 2] = m_1[:,:, 2] / norms[:,:]
    
    #TODO: Figure out a way to avoid this nasty for loop
    for i in range(m_1.shape[0]):
        for j in range(m_1.shape[1]):
            if m_1[i,j,0] == np.nan or m_1[i,j,1] == np.nan or m_1[i,j,2] == np.nan:
                 m_1[i,j,0] = 0.0
                 m_1[i,j,1] = 0.0
                 m_1[i,j,2] = 0.0
    return m_1

@nb.jit(cache = True, nopython = True,  fastmath = True)
def crossProduct(v1,v2):
    """Code to do a cross product in one dimension
    Args:
        v1 (ndarray): first vector (3d) 
        v2 (ndarray): second vector (3d)
    Returns:
        res (ndarray): Cross product- v1 x v2
    """
    res = np.zeros_like(v1)
    res[:,0] = v1[:,1]*v2[:,2] - v1[:,2]*v2[:,1]
    res[:,1] = v1[:,2]*v2[:,0] - v1[:,0]*v2[:,2]
    res[:,2] = v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
    return res

@nb.jit(cache = True, nopython = True,  fastmath = True)
def crossProduct2D(v1,v2):
    """Perform the crossproduct of all spins on a 2D system. The first two dimensions of the inputs are the lattice positions.
    At each  lattice position, this function performs the crossproduct between the vectors v1 and v2. 

    Args:
        v1 (ndarray(Nx,Ny,3)): first vector (3d) 
        v2 (ndarray(Nx,Ny,3)): second vector (3d)

    Returns:
        [ndarray]: [The result of performing the crossproduct between vectors v1 and v2 at all lattice points]
    """
    res = np.zeros_like(v1)
    res[:,:,0] = v1[:,:,1]*v2[:,:,2] - v1[:,:,2]*v2[:,:,1]
    res[:,:,1] = v1[:,:,2]*v2[:,:,0] - v1[:,:,0]*v2[:,:,2]
    res[:,:,2] = v1[:,:,0]*v2[:,:,1] - v1[:,:,1]*v2[:,:,0]
    return res

@nb.jit(cache = True, nopython = True,  fastmath = True)
def laplacian1D(mesh, border_size=0):
    """Calculates the laplacian in 1D using finite difference on 5 points. This is needed for the exchange energy-term of the hamiltonian. 

    Args:
        mesh (np.ndarray): The input is a time-part of the magentisation-matrix, so mesh is of size (N,3), ie N spins with all 3 components. 

    Returns:
        [np.array]: Returns the laplacian of the mesh-matrix. It is of the same dimension as the mesh-matrix. 
    """
    laplacian = np.zeros_like(mesh, dtype = dtype)
    const = 1/(12)
    N = mesh.shape[0]
    #Laplacian middle values
    laplacian[2:N-2, :] = (const)*(-mesh[0:N-4,:] +16*mesh[1:N-3,:] -30*mesh[2:N-2, :] + 16*mesh[3:N-1,:] - mesh[4:N,:])
    
    const2 = 12/25
    
    #Laplacian edge values(same as Alireza)
    #laplacian[0,:] = const2*(4*mesh[1,:] - 3*mesh[2,:] + (4/3)*mesh[3,:] - 0.25*mesh[4,:])
    #laplacian[N-1,:] = const2*(4*mesh[N-2,:]  - 3*mesh[N-3,:] + (4/3)*mesh[N-4,:] -0.25*mesh[N-5,:])
    
    ###Even Tries Neumann boundary conditions.
    #laplacian[0,:] = const*(-15*mesh[0,:] + 16*mesh[1,:] - mesh[2,:])
    
    #Next to edge (Same as Alireza)
    #laplacian[1,:] = const*(10*mesh[0,:] - 15*mesh[1,:] - 4*mesh[2,:] + 14*mesh[3,:] - 6*mesh[4,:] + 1*mesh[5,:])
    #laplacian[N-2,:] = const*(10*mesh[N-1,:] - 15*mesh[N-2,:] - 4*mesh[N-3,:] +14*mesh[N-3,:] -6*mesh[N-4,:] + mesh[N-5,:])
    
    ###Even Tries Neumann boundary conditions.
    #laplacian[1,:] = const*(15*mesh[0,:] -30*mesh[1, :] + 16*mesh[2,:] - mesh[3,:])
    
    laplacian[0,:] = -(3*mesh[0,:] - 4*mesh[1,:] + mesh[2,:])/2
    laplacian[1,:] = -(3*mesh[1,:] - 4*mesh[2,:] + mesh[3,:])/2
    
    laplacian[-2,:] = -(3*mesh[-2,:] - 4*mesh[-3,:] + mesh[-4,:])/2
    laplacian[-1,:] = -(3*mesh[-1,:] - 4*mesh[-2,:] + mesh[-3,:])/2

    ##Even tries to neumann
    #laplacian[-2,:] = const*(15*mesh[-1,:] -30*mesh[-2, :] + 16*mesh[-3,:] - mesh[-4,:])
    #laplacian[-1,:] = const*(-15*mesh[-1,:] + 16*mesh[-2,:] - mesh[-3,:])
    
    return laplacian[border_size:N-border_size]

@nb.jit(cache = True, nopython = True,  fastmath = True)
def laplacian2D(mesh,geom, border_size = 0):
    """Function to find laplacian to lowest order with Neumann boundary condition

    Args:
        mesh (_type_): _description_
        structure_arr (ndarray):
        border_size (int, optional): _description_. Defaults to 0.
        
    Returns:
        _type_: _description_
    """
    laplacian = np.zeros_like(mesh)
    
    
    
    ##Here edges are dealt with It's experimental
    #Lower edge
    laplacian[0,1:-1] = np.subtract(mesh[0,:-2] + mesh[0,2:] + mesh[1,1:-1], 3*mesh[0,1:-1])

    #Upper edge
    laplacian[-1,1:-1] = np.subtract(mesh[-1,:-2] + mesh[-1,2:] + mesh[-2,1:-1], 3*mesh[-1,1:-1])

    #Left edge
    laplacian[1:-1,0] = np.subtract(mesh[:-2,0] + mesh[2:,0] + mesh[1:-1,1], 3*mesh[1:-1,0])
    
    #right edge
    laplacian[1:-1,-1] = np.subtract(mesh[:-2,-1] + mesh[2:,-1] + mesh[1:-1,-2], 3*mesh[1:-1,-1])
    
    ##Here corners are dealt with
    #Lower left corner
    laplacian[0,0] = np.subtract(mesh[1,0] + mesh[0,1], 2*mesh[0,0])

    #Lower right corner
    laplacian[0,-1] = np.subtract(mesh[1,-1] + mesh[0,-2], 2*mesh[0,-1])

    #upper left corner
    laplacian[-1,0] = np.subtract(mesh[-2,0] + mesh[-1,1], 2*mesh[-1,0])

    #Upper right corner
    laplacian[-1,-1] = np.subtract(mesh[-2,-1] + mesh[-1,-2], 2*mesh[-1,-1])


    #Deal with geometry TODO:There is an ugly double for loop here, would be nice to find a way to drop it as it is quite slow. 
    #Boolean indexes are however not an option as numba does not support 2D boolean indexing. So no easy way of creating 
    #an array multiplication method as far as I see it. 
    coef = 4 
    for i in range(1,mesh.shape[0]-1):
        for j in range(1,mesh.shape[1]-1):
            coef = 4
            #Dealing with the bulk here, inside for to generalize for geometry
            if geom[i+1,j,0] == 0:
                coef -= 1
            if geom[i-1,j,0] == 0:
                coef -= 1
            if geom[i,j+1,0] == 0:
                coef -= 1
            if geom[i,j-1,0] == 0:
                coef -= 1
            laplacian[i,j,:] = np.subtract(mesh[i-1,j] + mesh[i+1,j] + mesh[i,j-1] + mesh[i,j+1],coef*mesh[i,j,:])


    return laplacian

@nb.jit(cache = True, nopython = True,  fastmath = True)
def laplacian2Dold(mesh,border_size =0):
    """Calculates the laplacian in 2D using finite difference on 9 surrounding points. This is needed for the exchange energy-term of the hamiltonian. 

    Args:
        mesh (np.ndarray): The input is a time-part of the spins-matrix, so mesh is of size (Nx,Ny,3), ie N spins with all 3 components. 

    Returns:
        [np.array]: Returns the laplacian of the mesh-matrix. It is of the same dimension as the mesh-matrix. 
    """ 
    laplacian = np.zeros_like(mesh, dtype = dtype)
    Nx = mesh.shape[0]
    Ny = mesh.shape[1]
    #This is Verena's version of the 2d laplace
    #######bulk : two up, two down, two left, two right
    #laplacian[2:Nx-2, 2:Ny-2,:] = -mesh[2:Nx-2,0:Ny-4,:] - mesh[2:Nx-2,1:Ny-3,:] - mesh[2:Nx-2,3:Ny-1,:] - mesh[2:Nx-2, 4:Ny,:]
    #+ 8*mesh[2:Nx-2, 2:Ny-2,:] - mesh[0:Nx-4, 2:Ny-2,:] - mesh[1:Nx-3, 2:Ny-2,:] - mesh[3:Nx-1,2:Ny-2,:] - mesh[4:Nx, 2:Ny-2,:] # ok

    laplacian[2:-2,2:-2,:] = np.subtract(8*mesh[2:-2,2:-2,:], (mesh[0:-4,2:-2,:]+mesh[1:-3,2:-2,:]+mesh[3:-1,2:-2,:]+mesh[4:,2:-2,:]+mesh[2:-2,0:-4,:]+mesh[2:-2,1:-3,:]+mesh[2:-2,3:-1,:]+mesh[2:-2,4:,:])) #bulk: two left, two right, two up, two down

    #[2:4] does not include 4!!!
    #print('bulk through')#,shape laplacian:',laplacian.shape)
    
    #left edge
    laplacian[2:-2,0,:] = np.subtract(8*mesh[2:-2,0,:], (mesh[0:-4,0,:]+mesh[1:-3,0,:]+mesh[3:-1,0,:]+mesh[4:,0,:]+mesh[2:-2,1,:]+mesh[2:-2,2,:]+mesh[2:-2,3,:]+mesh[2:-2,4,:])) # two up, two down, four right
    # second left edge
    laplacian[2:-2,1,:] = np.subtract(8*mesh[2:-2,1,:], (mesh[0:-4,0,:]+mesh[1:-3,0,:]+mesh[3:-1,0,:]+mesh[4:,0,:]+mesh[2:-2,0,:]+mesh[2:-2,2,:]+mesh[2:-2,3,:]+mesh[2:-2,4,:])) # two up, two down, one left, three right

    #upper
    laplacian[0,2:-2,:] = np.subtract(8*mesh[0,2:-2,:], (mesh[0,0:-4,:]+mesh[0,1:-3,:]+mesh[0,3:-1,:]+mesh[0,4:,:]+mesh[1,2:-2,:]+mesh[2,2:-2,:]+mesh[3,2:-2,:]+mesh[4,2:-2,:])) # two left, two right, four down
    # second upper edge
    laplacian[1,2:-2,:] = np.subtract(8*mesh[1,2:-2,:], (mesh[1,0:-4,:]+mesh[1,1:-3,:]+mesh[1,3:-1,:]+mesh[1,4:,:]+mesh[0,2:-2,:]+mesh[2,2:-2,:]+mesh[3,2:-2,:]+mesh[4,2:-2,:])) # two left, two right, one up, three down

    # right
    laplacian[2:-2,-1,:] = np.subtract(8*mesh[2:-2,-1,:], (mesh[0:-4,-1,:]+mesh[1:-3,-1,:]+mesh[3:-1,-1,:]+mesh[4:,-1,:]+mesh[2:-2,-2,:]+mesh[2:-2,-3,:]+mesh[2:-2,-4,:]+mesh[2:-2,-5,:])) # two up, two down, four left
    # second left edge
    laplacian[2:-2,-2,:] = np.subtract(8*mesh[2:-2,-2,:], (mesh[0:-4,-2,:]+mesh[1:-3,-2,:]+mesh[3:-1,-2,:]+mesh[4:,-2,:]+mesh[2:-2,-1,:]+mesh[2:-2,-3,:]+mesh[2:-2,-4,:]+mesh[2:-2,-5,:])) # two up, two down, one right, three left

    laplacian[-1,2:-2,:] = np.subtract(8*mesh[-1,2:-2,:], (mesh[-1,0:-4,:]+mesh[-1,1:-3,:]+mesh[-1,3:-1,:]+mesh[-1,4:,:]+mesh[-2,2:-2,:]+mesh[-3,2:-2,:]+mesh[-4,2:-2,:]+mesh[-5,2:-2,:])) # two left, two right, four down
    # second upper edge
    laplacian[-2,2:-2,:] = np.subtract(8*mesh[-2,2:-2,:], (mesh[-2,0:-4,:]+mesh[-2,1:-3,:]+mesh[-2,3:-1,:]+mesh[-2,4:,:]+mesh[-1,2:-2,:]+mesh[-3,2:-2,:]+mesh[-4,2:-2,:]+mesh[-5,2:-2,:])) # two left, two right, one up, three down
    
    
    #print('edges through')
    ####### corners

    #up left
    laplacian[0,0,:] = -mesh[1,0,:] -mesh[2,0,:] - mesh[3,0,:] -mesh[4,0,:]
    +8*mesh[0,0,:] -mesh[0,1,:] -mesh[0,2,:] -mesh[0,3,:] -mesh[0,4,:]

    laplacian[0,1,:] = -mesh[1,1,:] -mesh[2,1,:] - mesh[3,1,:] -mesh[4,1,:]
    +8*mesh[0,1,:] -mesh[0,0,:] -mesh[0,2,:] -mesh[0,3,:] -mesh[0,4,:]

    laplacian[1,0,:] = -mesh[0,0,:] -mesh[2,0,:] - mesh[3,0,:] -mesh[4,0,:]
    +8*mesh[1,0,:] -mesh[1,1,:] -mesh[1,2,:] -mesh[1,3,:] -mesh[1,4,:]

    laplacian[1,1,:] = -mesh[0,1,:] -mesh[2,1,:] - mesh[3,1,:] -mesh[4,1,:]
    +8*mesh[1,1,:] -mesh[1,0,:] -mesh[1,2,:] -mesh[1,3,:] -mesh[1,4,:]

    ##### up right
    laplacian[0,-2,:] = -mesh[1,-2,:] -mesh[2,-2,:] - mesh[3,-2,:] -mesh[4,-2,:]
    +8*mesh[0,-2,:] - mesh[0,-1,:] -mesh[0,-3,:] - mesh[0,-4,:] -mesh[0,-5,:]

    laplacian[0,-1,:] = -mesh[1,-1,:] -mesh[2,-1,:] - mesh[3,-1,:] -mesh[4,-1,:]
    +8*mesh[0,-1,:] - mesh[0,-2,:] -mesh[0,-3,:] - mesh[0,-4,:] -mesh[0,-5,:]

    laplacian[1,-1,:] = -mesh[0,-1,:] -mesh[2,-1,:] - mesh[3,-1,:] -mesh[4,-1,:]
    +8*mesh[1,-1,:] - mesh[1,-2,:] -mesh[1,-3,:] - mesh[1,-4,:] -mesh[1,-5,:]

    laplacian[1,-2,:] = -mesh[0,-2,:] -mesh[2,-2,:] - mesh[3,-3,:] -mesh[4,-4,:]
    +8*mesh[1,-2,:] - mesh[1,-1,:] -mesh[1,-3,:] - mesh[1,-4,:] -mesh[1,-5,:]

    #Lower left
    laplacian[-1,0,:] = - mesh[-2,0,:] -mesh[-3,0,:] - mesh[-4,0,:] -mesh[-5,0,:]
    +8*mesh[-1,0,:] - mesh[-1,1,:] -mesh[-1,2,:] - mesh[-1,3,:] -mesh[-1,4]

    laplacian[-2,0,:] = - mesh[-1,0,:] -mesh[-3,0,:] - mesh[-4,0,:] -mesh[-5,0,:]
    +8*mesh[-2,0,:] - mesh[-2,1,:] -mesh[-2,2,:] - mesh[-2,3,:] -mesh[-2,4,:]

    laplacian[-1,1,:] = - mesh[-2,1,:] -mesh[-3,1,:] - mesh[-4,1,:] -mesh[-5,1,:]
    +8*mesh[-1,1,:] - mesh[-1,0,:] -mesh[-1,2,:] - mesh[-1,3,:] -mesh[-1,4,:]

    laplacian[-2,1,:] = - mesh[-1,1,:] -mesh[-3,1,:] - mesh[-4,1,:] -mesh[-5,1,:]
    +8*mesh[-2,1,:] - mesh[-2,0,:] -mesh[-2,2,:] - mesh[-2,3,:] -mesh[-2,4,:]

    #lower right

    laplacian[-1,-1,:] = - mesh[-2,-1,:] -mesh[-3,-1,:] - mesh[-4,-1,:] -mesh[-5,-1,:]
    +8*mesh[-1,-1,:] - mesh[-1,-2,:]-mesh[-1,-3,:] - mesh[-1,-4,:] -mesh[-1,-5,:]

    laplacian[-1,-2,:] = - mesh[-2,-2,:] -mesh[-3,-2,:] - mesh[-4,-2,:] -mesh[-5,-2,:]
    +8*mesh[-1,-2,:] - mesh[-1,-1,:]-mesh[-1,-3,:] - mesh[-1,-4,:] -mesh[-1,-5,:]

    laplacian[-2,-1,:] = - mesh[-1,-1,:] -mesh[-3,-1,:] - mesh[-4,-1,:] -mesh[-5,-1,:]
    +8*mesh[-2,-1,:] - mesh[-2,-2,:]-mesh[-2,-3,:] - mesh[-2,-4,:] -mesh[-2,-5,:]
    
    laplacian[-2,-2,:] = - mesh[-1,-2,:] -mesh[-3,-2,:] - mesh[-4,-2,:] -mesh[-5,-2,:]
    +8*mesh[-2,-2,:] - mesh[-2,-1,:]-mesh[-2,-3,:] - mesh[-2,-4,:] -mesh[-2,-5,:]

    #print('corner through')


    
    return laplacian

@nb.jit(cache = True, nopython = True,  fastmath = True)
def exchangeEnergy(m_a_element, m_b_element, w_ex, A_1, A_2, border_size=0):
    """Calculates the exchange energy

    Args:
        m_a_element (np.ndarray): A time-component of one of the sub-lattices. It is of size (N,3)
        m_b_element (np.ndarray): A time-component of the other sub-lattice. It is of size (N,3)

    Returns:
        [type]: Returns a matrix of same size as the input-elements, where the exchange energies are calculated. 
    """
    N = m_a_element.shape[0]
    ex_array = np.zeros_like(m_a_element, dtype = dtype)
    
    ex_array[border_size:N-border_size] = -(w_ex*m_b_element[border_size:N-border_size] - A_1*laplacian1D(m_a_element, border_size=border_size) - A_2*laplacian1D(m_b_element, border_size=border_size))
    
    return ex_array            

@nb.jit(cache = True, nopython = True,  fastmath = True)
def exchangeEnergy2D(m_a_element, m_b_element, w_ex, A_1, A_2,border_size=0):
    """Function to calculate the exhcange energy in 2D

    Args:
        m_a_element (ndarray(Nx,Ny,3)): Time element of m_a
        m_b_element (ndarray(Nx,Ny,3)): Time element of m_b
        w_ex (float): omega exchange
        A_1 (float): Parameter
        A_2 (float): parameter
        border_size (int, optional): Size of border if it exists. Defaults to 0.

    Returns:
        The calculated exchange energy at timestep i from the given Hamiltonian: ndarray(Nx,Ny,3)
    """
    N_x = m_a_element.shape[0]
    N_y = m_a_element.shape[1]

    ex_array = np.zeros_like(m_a_element,dtype = dtype)
    ex_array = -(w_ex*m_b_element[:,:]) -A_1*laplacian2D(m_a_element, m_b_element) - A_2*laplacian2D(m_b_element,m_b_element)

    return ex_array

@nb.jit(cache = True, nopython = True,  fastmath = True)
def anisotropyEnergy1D(m_a, m_b, w_1, w_2, border_size=0):
    """Calculates the anisotropy-energy in the 1D case 

    Args:
        m_a (np.ndarray): A time-element of the spins-array of size (N,3)
        m_b (np.ndarray): A time-element of the spins-array of size (N,3)

    Returns:
        [type]: Returns the calculated anisotropy
    """
    ani_array = np.zeros_like(m_a, dtype = dtype)
    ani_array = -(w_1 @ m_a.T).T - (w_2 @ m_b.T).T
    return ani_array

@nb.jit(cache = True, nopython = True,  fastmath = True)
def anisotropyEnergy2D(m_a, m_b, w_1, w_2, border_size=0):
    """Calculates the anisotropy energy in the 2D case

    Args:
        m_a ([np.ndarray(Nx,Ny,3)]): A time-element of the spin-array of size (N,3)
        m_b ([np.ndarray(Nx,Ny,3)]): A time-element of the spin-array of size (N,3)
        w_1 ([np.ndarray(3,3)]): [The first term anisotropy tensor]
        w_2 ([np.ndarray(3,3)]): [The second term anisotropy tensor]
        border_size (int, optional): [description]. Defaults to 0.

    Returns:
        [np.ndarray(Nx,Ny,3)]: Resulting array after the anisotrpy calculations
    """
    ani_array = np.zeros_like(m_a, dtype = dtype)
    """
    Nx = m_a.shape[0]
    Ny = m_a.shape[1]
    w_1_calc = np.zeros((Nx,Ny, 3,3),dtype = dtype)
    w_2_calc = np.zeros_like(w_1_calc)
    w_1_calc[:,:] = w_1 
    w_2_calc[:,:] = w_2   """
    
    res_1_x = w_1[:,:,0,0]*m_a[:,:,0] + w_1[:,:,0,1]*m_a[:,:,1] + w_1[:,:,0,2]*m_a[:,:,2]
    res_1_y = w_1[:,:,1,0]*m_a[:,:,0] + w_1[:,:,1,1]*m_a[:,:,1] + w_1[:,:,1,2]*m_a[:,:,2]
    res_1_z = w_1[:,:,2,0]*m_a[:,:,0] + w_1[:,:,2,1]*m_a[:,:,1] + w_1[:,:,2,2]*m_a[:,:,2]
    
    res_2_x = w_2[:,:,0,0]*m_b[:,:,0] + w_2[:,:,0,1]*m_b[:,:,1] + w_2[:,:,0,2]*m_b[:,:,2]
    res_2_y = w_2[:,:,1,0]*m_b[:,:,0] + w_2[:,:,1,1]*m_b[:,:,1] + w_2[:,:,1,2]*m_b[:,:,2]
    res_2_z = w_2[:,:,2,0]*m_b[:,:,0] + w_2[:,:,2,1]*m_b[:,:,1] + w_2[:,:,2,2]*m_b[:,:,2]
    
    
    ani_array[:,:,0] = -res_1_x - res_2_x
    ani_array[:,:,1] = -res_1_y - res_2_y
    ani_array[:,:,2] = -res_1_z - res_2_z

    return ani_array

@nb.jit(cache = True, nopython = True,  fastmath = True)
def zeemanEnergy(h,i, border_size=0):
    """Calculates the zeeman energy

    Args:
        h (np.ndarray(Nx,Nt,3)): The magnetic field at all lattice positions, times and directions.
        i (int): the time-index. The point of this i is that the magnetic field can both be spatially and be temporally dependent,
        in order to use the correct magnetif field, i is needed for it to be accessed. 

    Returns:
        [np.ndarray]: Returns the calculated zeeman energy vector for each lattice point, thus the size is (N,3)
    """
    zeeman_arr = np.zeros_like(h[:,0,:], dtype =dtype)
    zeeman_arr[border_size:h.shape[0]-border_size] = h[border_size:h.shape[0]-border_size,i,:]
    return zeeman_arr

@nb.jit(cache = True, nopython = True,  fastmath = True)
def zeemanEnergy2D(h,i, border_size=0):
    """Calculates the zeeman energy

    Args:
        h (np.ndarray(Nx,Ny,Nt,3)): The magnetic field at all lattice positions, times and directions.
        i (int): the time-index. The point of this i is that the magnetic field can both be spatially and be temporally dependent,
        in order to use the correct magnetif field, i is needed for it to be accessed. 

    Returns:
        [np.ndarray]: Returns the calculated zeeman energy vector for each lattice point, thus the size is (N,3)
    """
    zeeman_arr = np.zeros_like(h[:,:,0,:], dtype =dtype)
    Nx = zeeman_arr.shape[0]
    Ny = zeeman_arr.shape[1]
    zeeman_arr[border_size:h.shape[0]-border_size,:,:] = h[border_size:h.shape[0]-border_size,:,i,:]
    return zeeman_arr

@nb.jit(cache = True, nopython = True,  fastmath = True)
def DMIEnergy(m, a, d, border_size=0):
    """Calculates the DMI-energy

    Args:
        m (np.ndarray): m_matrix
        a (np.array): direction of the DMI-term

    Returns:
        [np.ndarray]: The calculated DMI-array
    """
    d_calc = np.zeros_like(m)
    d_calc[:] = d
    if a:
        DMI_arr = -crossProduct(m,d_calc)
    else:
        DMI_arr = crossProduct(m,d_calc)
    return DMI_arr

@nb.jit(cache = True, nopython = True,  fastmath = True)
def DMIEnergy2D(m,a,d,border_size = 0):
    """Calculates the DMI-energy

    Args:
        m (np.ndarray(Nx,Ny,3)): m_matrix
        a (bool): a boolean switch that is used to seperte the DMI contribution from m of type m_a and m_b since their sign contributions are opposite
        d (np.array(3)): direction of the DMI-term

    Returns:
        [np.ndarray]: The calculated DMI-array
    """
    d_calc = np.zeros_like(m)
    d_calc[:,:] = d
    Nx = m.shape[0]
    Ny = m.shape[1]
    if a:
        DMI_arr = -crossProduct2D(m,d_calc)
    else:
        DMI_arr = crossProduct2D(m,d_calc)
    return DMI_arr

@nb.jit(cache = True, nopython = True,  fastmath = True)
def secondDMI2D(m,border_size=0):
    """Performs the cross-product-part of the DMI calculations using dx = dy = dz = 1 due to the non-dimensionality of the equations
    This is currently written using the central stencil using a point to the left and right of the point in question.

    Args:
        m ([np.ndarray(Nx,Ny,3)]): [The entire grid of spins at time i]
        border_size ([int], optional): [The size of the borders where the spins are fixed]. Defaults to border_size.

    Returns:
        [np.ndarray(Nx,Ny,3)]: [Returns the curl of the spins at time i]
    """
    result = np.zeros_like(m)
    Nx = m.shape[0]
    Ny = m.shape[1]

    result[:,:,0] = PartialY_2D(m,2,Ny)
    result[:,:,1] = -1*PartialX_2D(m,2,Nx)
    result[:,:,2] = PartialX_2D(m,1,Nx) - PartialY_2D(m,0,Ny)
    return result

@nb.jit(cache = True, nopython = True,  fastmath = True)
def PartialX_2D(m,mcomponent,Nx):
    """Calculates the partial derivative in the x-direction for the spin-component mcomponent

    Args:
        m ([np.ndarray(Nx,Ny,3)]): [The entire grid of spins at time i]
        mcomponent ([int]): [The spin-component that is to be derivated. 0 = x, 1 = y, 2 = z]
        Nx ([int]): [number of lattice positions]

    Returns:
        [type]: [Returns the  partial derivative in the x-direction of all the spins at the mcomponent'th spincomponent]
    """
    const = 1/(12)
    result = np.zeros_like(m[:,:,0])
    result[2:Nx-2,:] = (const)*(m[0:Nx-4,:,mcomponent] -8*m[1:Nx-3,:,mcomponent] +8*m[3:Nx-1,:,mcomponent] - m[4:Nx,:,mcomponent])
    result[0,:] = (const)*(-25*m[0,:,mcomponent] +48*m[1,:,mcomponent] -36*m[2,:,mcomponent] + 16*m[3,:,mcomponent] - 3*m[4,:,mcomponent])
    result[1,:] = (const)*(-3*m[0,:,mcomponent] -10*m[1,:,mcomponent] +18*m[2,:,mcomponent] -6*m[3,:,mcomponent] +1*m[4,:,mcomponent])
    result[-2,:] = (const)*(-1*m[-5,:,mcomponent] + 6*m[-4,:,mcomponent]-18*m[-3,:,mcomponent]+10*m[-2,:,mcomponent]+3*m[-1,:,mcomponent])
    result[-1,:] = (const)*(3*m[-5,:,mcomponent]-16*m[-4,:,mcomponent]+36*m[-3,:,mcomponent]-48*m[-2,:,mcomponent]+25*m[-1,:,mcomponent])
    return result

@nb.jit(cache = True, nopython = True,  fastmath = True)
def PartialY_2D(m,mcomponent,Ny):
    """Calculates the partial derivative in the y-direction for the spin-component mcomponent

    Args:
        m ([np.ndarray(Nx,Ny,3)]): [The entire grid of spins at time i]
        mcomponent ([int]): [The spin-component that is to be derivated. 0 = x, 1 = y, 2 = z]
        Nx ([int]): [number of lattice positions]

    Returns:
        [type]: [Returns the  partial derivative in the y-direction of all the spins at the mcomponent'th spincomponent]
    """
    const = 1/(12)
    result = np.zeros_like(m[:,:,0])
    result[:,2:Ny-2] = (const)*(m[:,0:Ny-4,mcomponent] -8*m[:,1:Ny-3,mcomponent] +8*m[:,3:Ny-1,mcomponent] - m[:,4:Ny,mcomponent])
    result[:,0] = (const)*(-25*m[:,0,mcomponent] +48*m[:,1,mcomponent] -36*m[:,2,mcomponent] + 16*m[:,3,mcomponent] - 3*m[:,4,mcomponent])
    result[:,1] = (const)*(-3*m[:,0,mcomponent] -10*m[:,1,mcomponent] +18*m[:,2,mcomponent] -6*m[:,3,mcomponent] +1*m[:,4,mcomponent])
    result[:,-2] = (const)*(-1*m[:,-5,mcomponent] + 6*m[:,-4,mcomponent]-18*m[:,-3,mcomponent]+10*m[:,-2,mcomponent]+3*m[:,-1,mcomponent])
    result[:,-1] = (const)*(3*m[:,-5,mcomponent]-16*m[:,-4,mcomponent]+36*m[:,-3,mcomponent]-48*m[:,-2,mcomponent]+25*m[:,-1,mcomponent])
    return result

@nb.jit(cache = True, nopython = True,  fastmath = True)
def secondDMI1D(m,border_size=0):
    """Performs the cross-product-part of the DMI calculations using dx = dy = dz = 1 due to the non-dimensionality of the equations
    This is currently written using the central stencil using a point to the left and right of the point in question.

    Args:
        m ([np.ndarray(Nx,Ny,Nz,3)]): [The entire grid of spins at time i]
        border_size ([int], optional): [The size of the borders where the spins are fixed]. Defaults to border_size.

    Returns:
        [type]: [description]
    """
    result = np.zeros_like(m)
    const = 1/(12)
    N = m.shape[0]
    #y-direction
    result[2:N-2,1] = (const)*(m[0:N-4,2] -8*m[1:N-3,2] +8*m[3:N-1,2] - m[4:N,2])
    result[0,1] = (const)*(-25*m[0,2] + 48*m[1,2] - 36*m[2,2] + 16*m[3,2] - 3*m[4,2])
    result[1,1] = (const)*(-3*m[0,2] - 10*m[1,2] + 18*m[2,2] - 6*m[3,2] + m[4,2])
    result[-2,1] = (const)*(-m[-5,2] + 6*m[-4,2] - 18*m[-3,2] + 10*m[-2,2] + 3*m[-1,2])
    result[-1,1] = (const)*(3*m[-5,2] - 16*m[-4,2] + 36*m[-3,2] - 48*m[-2,2] + 25*m[-1,2])

    #z-direction
    result[2:N-2,2] = (const)*(m[0:N-4,1] -8*m[1:N-3,1] +8*m[3:N-1,1] - m[4:N,1])
    result[0,2] = (const)*(-25*m[0,1] + 48*m[1,1] - 36*m[2,1] + 16*m[3,1] - 3*m[4,1])
    result[1,2] = (const)*(-3*m[0,1] - 10*m[1,1] + 18*m[2,1] - 6*m[3,1] + m[4,1])
    result[-2,2] = (const)*(-m[-5,1] + 6*m[-4,1] - 18*m[-3,1] + 10*m[-2,1] + 3*m[-1,1])
    result[-1,2] = (const)*(3*m[-5,1] - 16*m[-4,1] + 36*m[-3,1] - 48*m[-2,1] + 25*m[-1,1])
    return result

@nb.jit(cache = True, nopython = True,  fastmath = True)
def f(m_1,m_2,i, a, geom, init = False, border_size=0, constants = ()):
    """Calculates mdot, i.e f(m_1,m_2,i) which is used in the RK4 routine. 

    Args:
        m_1 (np.ndarray(Nx,3)): a time-instance of the m_a spins-matrix of size (N,3)
        m_2 ([np.ndarray(Nx,3)]): a time-instance of the m_b spins-matrix of size (N,3)
        i ([int]): a time-instance of the simulation
        a (bool): a boolean switch that is used to seperte the DMI contribution from m of type m_a and m_b since their sign contributions are opposite
        constants ([tuple]): All the initialized compoenents needed for the terms of the simulation
    Returns:
        [np.ndarray(Nx,3)]: Returns mdot, a (N,3) ndarray 
                Returns the current at this time step given my m cross m_dot
    """
    #Load constants
    w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J_func, J_args,d,h_func, h_args,D, Temp = constants 
    H_T = Temp
    H_T = np.multiply(np.random.randn(*H_T.shape),Temp)
    #Find effective field derivative
    H_1 = np.zeros_like(m_1, dtype = dtype)
    H_1 = exchangeEnergy(m_1, m_2,w_ex,A_1, A_2, border_size=border_size)  + h_func(m_1,i,h_args) + anisotropyEnergy1D(m_1, m_2,w_1,w_2,border_size=border_size) + DMIEnergy(m_2, a, d) + D*secondDMI1D(m_1, border_size=border_size) + H_T
    #Find constants for later use
    K = np.zeros_like(m_1, dtype = dtype)
    K = 1/(1+alpha**2)
    beta = beta/(K) 

    #Calculate the actual 
    f_1 = np.zeros_like(m_1, dtype = dtype)
    f_1 = -(K)*crossProduct(m_1, H_1 + C*J_func(m_1,i,J_args) + alpha[:,:]*(crossProduct(m_1, H_1 +beta*J_func(m_1,i,J_args))))
    return f_1, f_1

@nb.jit(cache = True, nopython = True,  fastmath = True)
def f2D(m_1,m_2,i, a, geom, init = False, border_size=0, constants = ()):
    """[Calculates mdot, i.e f(m_1,m_2,i) which is used in the RK4 routine]

    Args:
        m_2 ([np.ndarray(Nx,Ny,3)]): [description]
        i ([int]): a time-instance of the simulation
        a (bool): a boolean switch that is used to seperte the DMI contribution from m of type m_a and m_b since their sign contributions are opposite
        constants ([tuple]): All the initialized components needed for the terms of the simulation

    Returns:
        [np.ndarray(Nx,Ny,3)]: [returns mdot, a (Nx,Ny,3) sized array]
    """
    #Load constants
    w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J_func, J_args,d,h_func, h_args,D, Temp = constants 
    H_T = Temp
    H_T= np.multiply(np.random.randn(*H_T.shape),Temp) 
    
    #Find effective field derivative H
    H_1 = np.zeros_like(m_1, dtype = dtype)
    H_1 = exchangeEnergy2D(m_1, m_2,w_ex,A_1, A_2, border_size=border_size) + anisotropyEnergy2D(m_1,m_2, w_1, w_2, border_size = border_size) + h_func(m_1,i,h_args) + DMIEnergy2D(m_2,a,d) + H_T + D*secondDMI2D(m_1, border_size = border_size) 
    
    #Find constants for later use
    K = np.zeros_like(m_1, dtype = dtype)
    K = 1/(1+alpha**2)
    beta = beta/(K) 
    
    #Calculate the actual time derivative
    f_1 = np.zeros_like(m_1, dtype = dtype)

    f_1[border_size:f_1.shape[0]-border_size,border_size:f_1.shape[1]-border_size] = -(K[border_size:f_1.shape[0]-border_size,border_size:f_1.shape[1]-border_size])*crossProduct2D(m_1[border_size:f_1.shape[0]-border_size,border_size:f_1.shape[1]-border_size], H_1[border_size:f_1.shape[0]-border_size,border_size:f_1.shape[1]-border_size] + C*J_func(m_1,i,J_args)[border_size:f_1.shape[0]-border_size,border_size:f_1.shape[1]-border_size] + alpha[border_size:f_1.shape[0]-border_size,border_size:f_1.shape[1]-border_size]*(crossProduct2D(m_1[border_size:f_1.shape[0]-border_size,border_size:f_1.shape[1]-border_size], H_1[border_size:f_1.shape[0]-border_size,border_size:f_1.shape[1]-border_size] +beta[border_size:f_1.shape[0]-border_size,border_size:f_1.shape[1]-border_size]*J_func(m_1,i,J_args)[border_size:f_1.shape[0]-border_size,border_size:f_1.shape[1]-border_size])))
    #f_1 = -K*(crossProduct2D(m_1, H_1 + C*J_func(m_1,i,J_args) + alpha*(crossProduct2D(m_1, H_1 + beta*J_func(m_1,i,J_args)))))
    return f_1, f_1

@nb.jit(cache = True, nopython = True,  fastmath = True)
def timeStep(m_1, m_2, dt, geom,i, a, border_size=0, constants = ()):
    """A timestep of the RK4 routine. Uses f(m_1,m_2,i) to calculate a timestep

    Args:
        m_1 (np.ndarray(Nx,3)): m_1 is either m_a or m_b, depending on wether this function is called to calculate m_a dot or m_b dot respectively
        m_2 (np.ndarray(Nx,3)): the other spin-sublattice
        dt (float): a fixed timetep, used in RK4
        i (int): The i'th time
        a (bool): a boolean switch that is used to seperte the DMI contribution from m of type m_a and m_b since their sign contributions are opposite
        constants (tuple): All the initialized components needed for the terms of the simulation
    Returns:
        [np.ndarray(Nx,3)]: [A timestep after using the runge kutta routine]
    """
    k1,mdot = f(m_1[:], m_2[:],i,a, geom, border_size=border_size, constants = constants)
    
    k2,_ = f(np.add(m_1[:, :], k1*0.5*dt), m_2[:], i,a, geom, border_size=border_size, constants = constants)
    
    k3,_ = f(np.add(m_1[:, :], 0.5*k2*dt), m_2[:], i,a, geom, border_size=border_size, constants = constants)
    
    k4,_ = f(np.add(m_1[:, :], dt*k3), m_2[:], i,a, geom, border_size=border_size, constants = constants)
    
    m_1[:] += (dt / 6) * (k1 + 2* k2 + 2 * k3 + k4)

    m_1 = normalize1D(m_1)    
    return m_1,mdot

@nb.jit(cache = True, nopython = True,  fastmath = True)
def timeStep2D(m_1, m_2, dt, geom, i, a, border_size=0, constants = ()):
    """A timestep of the RK4 routine. Uses f(m_1,m_2,i) to calculate a timestep for a 2D system

    Args:
        m_1 ([np.ndarray(Nx,Ny,3)]): m_1 is either m_a or m_b, depending on wether this function is called to calculate m_a dot or m_b dot respectively
        m_2 ([np.ndarray(Nx,Ny,3)]): the other spin-sublattice
        dt ([float]): a fixed timetep, used in RK4
        i ([int]): The i'th time
        a ([bool]): a boolean switch that is used to seperte the DMI contribution from m of type m_a and m_b since their sign contributions are opposite
        border_size (int, optional): [description]. Defaults to 0.
        constants (tuple, optional): All the initialized components needed for the terms of the simulation. Defaults to ().

    Returns:
        [np.ndarray(Nx,Ny,3)]: [A timestep after using the runge kutta routine]
    """
    k1,mdot = f2D(m_1, m_2,i,a, geom, border_size=border_size, constants = constants)
    
    k2,_ = f2D(np.add(m_1, k1*0.5*dt), m_2, i,a, geom, border_size=border_size, constants = constants)
    
    k3,_ = f2D(np.add(m_1, 0.5*k2*dt), m_2, i,a, geom, border_size=border_size, constants = constants)
    
    k4,_ = f2D(np.add(m_1, dt*k3), m_2, i,a, geom, border_size=border_size, constants = constants)
    
    m_1[:] += (dt / 6) * (k1 + 2* k2 + 2 * k3 + k4)

    m_1 = normalize2D(m_1,geom) #TODO: Might need to implement a nan config thing here, think might be causing nan values

    
    return m_1,mdot

def timeEvolution(m_a, m_b, N_steps, dt, geom, cut_borders_at = 0, border_size = 0, stride = 0, constants = ()):
    """Calculates the state of both sublattice spins m_a, m_b for all timesteps N_steps using the RK45 routine.

    Args:
        m_a (np.ndarray(N,N_steps,3)): the m_a sublattice spins matrix of size (N,N_steps,3)
        m_b (np.ndarray(N,N_steps,3)): the m_b sublattice spins matrix of size (N,N_steps,3)
        N_steps (int): The number of iterations to perform using a fixed timestep
        dt (float): One time-step increment
        stride (int, optional): The number of timesteps that are skipped before the state of the system is saved. Defaults to 0.
        constants (tuple, optional): All the initialized components needed for the terms of the simulation. Defaults to ().

    Returns:
        m_a (np.ndarray(N,N_steps,3)): the m_a sublattice spins matrix of size (N,N_steps,3), now with the spin-matrices updated for all times
        m_b (np.ndarray(N,N_steps,3)): the m_b sublattice spins matrix of size (N,N_steps,3), now with the spin-matrices updated for all times
        T (np.ndarray(N_steps)): This is a numpy array with all the times of the simulation
    """
    T = np.zeros(int(N_steps/stride), dtype = dtype)
    mdot_a = np.zeros((np.shape(m_a)), dtype = dtype)
    mdot_b = np.zeros((np.shape(m_b)), dtype = dtype)
    j = 0
    temp_m_a = np.zeros((m_a[:,0,:].shape), dtype = dtype)
    temp_m_b = np.zeros_like(temp_m_a)
    temp_m_a_dot = np.zeros_like(temp_m_a)
    temp_m_b_dot = np.zeros_like(temp_m_a)
    inter_m_a = np.zeros_like(temp_m_a)
    temp_m_a = m_a[:,0,:]
    temp_m_b = m_b[:,0,:]
    for i in tqdm(range(N_steps)):
        if i == cut_borders_at:
            border_size = 0
        inter_m_a = temp_m_a
        temp_m_a, temp_m_a_dot= timeStep(temp_m_a, temp_m_b, dt,geom, i, True, border_size=border_size, constants = constants)
        temp_m_b, temp_m_b_dot= timeStep(temp_m_b, inter_m_a, dt,geom, i, False,border_size=border_size, constants = constants)
        if i%stride == 0:
            T[j] += (i)*dt
            m_a[:,j,:] = temp_m_a
            mdot_a[:,j,:] = temp_m_a_dot
            m_b[:,j,:] = temp_m_b
            mdot_b[:,j,:] = temp_m_b_dot
            j += 1
    return m_a, m_b, T,mdot_a,mdot_b

def timeEvolution2D(m_a, m_b, N_steps, dt, geom, cut_borders_at = 0, border_size = 0, stride = 0, constants = ()):
    T = np.zeros(N_steps, dtype = dtype)
    mdot_a = np.zeros((np.shape(m_a)), dtype = dtype)
    mdot_b = np.zeros((np.shape(m_b)), dtype = dtype)
    j = 0
    temp_m_a = np.zeros((m_a[:,:,0,:].shape), dtype = dtype)
    temp_m_b = np.zeros_like(temp_m_a)
    temp_m_a_dot = np.zeros_like(temp_m_a)
    temp_m_b_dot = np.zeros_like(temp_m_a)
    inter_m_a = np.zeros_like(temp_m_a)
    temp_m_a = m_a[:,:,0,:]
    temp_m_b = m_b[:,:,0,:]

    #Extract border array
    """border = np.diff(geom[:,:,0],axis = 0, n=2)[:,1:-1] + np.diff(geom[:,:,0], axis = 1, n= 2)[1:-1,:]
    border[border > 0] = 0.0
    border[border < 0] = 1.0
    """


    for i in tqdm(range(N_steps)):
        if i == cut_borders_at:
            border_size = 0

        inter_m_a = temp_m_a
        temp_m_a, temp_m_a_dot = timeStep2D(temp_m_a, temp_m_b, dt, geom, i, True, border_size=border_size, constants = constants)
        temp_m_b, temp_m_b_dot = timeStep2D(temp_m_b, inter_m_a, dt, geom, i, False,border_size=border_size, constants = constants)
        
        if (i)%stride == 0:
            T[j] += (i)*dt
            m_a[:,:,j,:] = temp_m_a
            mdot_a[:,:,j,:] = temp_m_a_dot
            m_b[:,:,j,:] = temp_m_b
            mdot_b[:,:,j,:] = temp_m_b_dot
            j += 1
    return m_a, m_b, T,mdot_a,mdot_b

def solver1D(N_steps, dt, init_state, geom, cut_borders_at = 0, border_size = 0, stride = 0, constants = ()):
    """A wrapper function that is needed to call to both initialize the system as well as solve it. 

    Args:
        N (int): Number of lattice sites /spins
        N_steps (int): The number of time-steps used in the RK45 routine
        dt (float): the timestep used in RK4
        initialiser (function): A potentially custom function that is used to initalize the spins. 
        initargs (tuple, optional): [description]. Defaults to (). A tuple of the necessary arguments for the initializer function. 

    Returns:
        m_a (np.ndarray(N,N_steps,3)): All the spins after the simulation on the a-sublattice
        m_b (np.ndarray(N,N_steps,3)): All the spins after the simulation on the b-sublattice
        T (np.ndarray(N_steps)): A vector of all the times in the simulation
    """
    m_a_start,m_b_start = init_state[0], init_state[1]    
    m_a, m_b, T,mdot_a,mdot_b = timeEvolution(m_a_start, m_b_start, N_steps, dt, geom, cut_borders_at = cut_borders_at, border_size = border_size, constants = constants, stride = stride)
    return m_a, m_b, T, mdot_a, mdot_b

def solver2D(N_steps, dt, init_state, geom, cut_borders_at = 0, border_size = 0, stride = 0, constants = ()):
    """A wrapper function that is needed to call to both initialize the system as well as solve it

    Args:
        N_steps ([int]): [Number of timesteps]
        dt ([float]): [size of timestep used in rk4 method]
        init_state ([np.ndarray(Nx,Ny,N_steps,3)]): 2 arrays of size (Nx,Ny,N_steps,3) where the first time-element represents the initial state of m_a and m_b respectively
        cut_borders_at (int, optional): [description]. Defaults to 0.
        border_size (int, optional): [size of the border of the system where the spins remain fixed]. Defaults to 0.
        stride (int, optional): The number of timesteps that are skipped before the state of the system is saved. Defaults to 0.
        constants (tuple, optional): All the initialized components needed for the terms of the simulation. Defaults to ().

    Returns:
        m_a ([np.ndarray(Nx,Ny,N_steps//stride,3)]): The spins at the A-sublattice at all times
        m_b ([np.ndarray(Nx,Ny,N_steps//stride,3)]): The spins at the B-sublattice at all times
        T ([np.ndarray(N_steps//stride)]): The times of the simulated system
        mdot_a ([np.ndarray(Nx,Ny,N_steps//stride,3)]): The timederivative of the spins at the A-sublattice at all times
        mdot_b ([np.ndarray(Nx,Ny,N_steps//stride,3)]): The timederivative of the spins at the B-sublattice at all times
    """
    m_a_start, m_b_start = init_state[0], init_state[1]
    m_a, m_b, T,mdot_a,mdot_b = timeEvolution2D(m_a_start, m_b_start, N_steps, dt, geom, cut_borders_at = cut_borders_at, border_size = border_size,stride = stride, constants = constants)
    return m_a, m_b, T, mdot_a, mdot_b