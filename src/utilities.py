import random
import numpy as np
import config 
import RK4
import matplotlib.pyplot as plt
import plotting as plot
import numba as nb
from joblib import Parallel, delayed
import JFunctions as Jfunc
import hFunctions as hfunc
import imageToNdarray as imToArr
plt.switch_backend('Qt5Agg')
#INITIALIZATION OF DIFFERENT MAGNETIC FIELDS

#If you want to create your own custom magnetic field, you may do so by creating a function that
#returns a matrix of correct shape. The shape of the magnetic field is 
# (Nx,Ny,Nz,N_steps,3) for the 3D case 
# (Nx,Ny,N_steps,3) for the 2D case
# (Nx,N_steps,3) for the 1D case
# From there, call on the magnetic field at the initialization part of the main file. 

def flopH(N,N_steps,Ny = 0, Nz = 0, on = True,dir = 1, hargs = ()):
    """This is a way to activate the magnetic field after some time. This particular function creates 
    a magnetic field after the 5000'th timestep weak enough to create spin-flip effect

    Args:
        N (int): Number of spins
        N_steps (int): Number of time-steps of the simulation
        on (bool, optional): [description]. Putting this to true activates the magnetic field. If you don't want this effect,
        one can simply put it to false, so that there is no magnetic field in the system. Defaults to True.
        dir (int,optinal): The direction of where you want the magnetic field to point. x-direction = 0, y-direction = 1, z-direction = 2. Defaults to 0, ie x-direction
    Returns:
        h (np.ndarray(N,N_steps,3): returns the magnetic field vector for all timesteps for all the spins in all direction, ie
        it is of size (N,N_steps,3) 
    """
    if Ny == 0:
        h = np.zeros((N,N_steps,3), dtype = config.dtype)
        if on:
            #h[:,:, dir] = np.sqrt(config.w_hc)
            h[:,:, dir] = 0.97*np.sqrt(2.4*10**-2*(2*config.w_ex + 2.4*10**-2))
            #h[:, 5000:, 1] = 0.1*np.pi*np.sqrt(2.4*10**-2*(2*config.w_ex + 2.4*10**-2))
    elif Ny != 0 and Nz ==0:
        h = np.zeros((N,Ny,N_steps,3))
        if on:
            h[:,:,:, dir] = 0.97*np.sqrt(2.4*10**-2*(2*config.w_ex + 2.4*10**-2))
    print("square root spin flop" , np.sqrt(2.4*10**-2*(2*config.w_ex + 2.4*10**-2)))
    print("Current B-field", h[0,0, dir])
    print("exchange: ", config.w_ex)
    print("A_1", config.A_1)
    return h

def flipH(N,N_steps,Ny = 0, Nz = 0,on = True,dir = 0, hargs = ()):
    """This is a way to activate the magnetic field after some time. This particular function creates 
    a magnetic field after the 15 000'th timestep so strong that it creates spin-flop

    Args:
        N (int): Number of spins
        N_steps (int): Number of time-steps of the simulation
        on (bool, optional): [description]. Putting this to true activates the magnetic field. If you don't want this effect,
        one can simply put it to false, so that there is no magnetic field in the system. Defaults to True.
        dir (int,optinal): The direction of where you want the magnetic field to point. x-direction = 0, y-direction = 1, z-direction = 2. Defaults to 0, ie x-direction
    Returns:
        h (np.ndarray(N,N_steps,3): returns the magnetic field vector for all timesteps for all the spins in all direction, ie
        it is of size (N,N_steps,3) 
    """
    if Ny == 0 and Nz == 0:
        h = np.zeros((N,N_steps,3), dtype = config.dtype)
        if on:
            h[:, 0:, dir] = 100*np.pi*np.sqrt(2.4e-2*config.A_1)
    elif Ny != 0 and Nz == 0:
        h = np.zeros((N,Ny,N_steps,3), dtype = config.dtype)
        if on:
            h[:,:,0:, dir] = 100*np.pi*np.sqrt(2.4e-2*config.A_1)
    return h

def setHField(N,N_steps,Ny = 0, Nz = 0, on = True, dir = 1, hargs = ()):
    """Create a magnetic field field that remains constant at all times in a certain direciton given by dir

    Args:
        N ([int]): Size of the system
        N_steps ([int]): Numer of timesteps of the simulation
        Ny (int, optional): System size in the y-direction. If set to 0, it returns the field for every spin for a 1D system. Defaults to 0.
        Nz (int, optional): System size in the z-direction. Defaults to 0.
        on (bool, optional): Turn the magnetic field on or off. Defaults to True.
        dir (int, optional): The direction of the constant field. 0  = x, 1 = y, 2 = z. Defaults to 1.
        hargs (tuple, optional): A tuple that contains the strength of the magnetic field. Defaults to ().

    Returns:
        [type]: [description]
    """
    strength = hargs
    print("Sat field to strength: ", strength)
    if Ny == 0 and Nz == 0:
        h = np.zeros((N,N_steps,3), dtype = config.dtype)
        if on:
            h[:, :, dir] = strength
    elif Ny != 0 and Nz == 0:
        h = np.zeros((N,Ny,N_steps,3), dtype = config.dtype)
        if on:
            h[:,:, :, dir] = strength
    return h

def constantLowH(N,N_steps,Ny = 0, Nz = 0, on = False,dir = 0, hargs = (0,0,0,0,0,0)):
    """This is a way to activate the magnetic field after some time. This particular function creates 
    a magnetic field after the 15000'th timestep that is weaker than spin-flip, so you get a constant
    relatively low magnetif field 

    Args:
        N (int): Number of spins
        N_steps (int): Number of time-steps of the simulation
        on (bool, optional): Putting this to true activates the magnetic field. If you don't want this effect,
        one can simply put it to false, so that there is no magnetic field in the system. Defaults to True.
        dir (int,optinal): The direction of where you want the magnetic field to point. x-direction = 0, y-direction = 1, z-direction = 2. Defaults to 0, ie x-direction

    Returns:
        h (np.ndarray(N,N_steps,3): returns the magnetic field vector for all timesteps for all the spins in all direction, ie
        it is of size (N,N_steps,3) 
    """
    if Ny == 0 and Nz == 0:
        h = np.zeros((N,N_steps,3), dtype = config.dtype)
        if on:
            h[:, :, dir] = np.pi*np.sqrt(5.4*10e-2*config.A_1)
    elif Nz == 0 and Ny != 0:
        h = np.zeros((N,Ny,N_steps,3), dtype = config.dtype)
        if on:
            h[:,:, :, dir] = np.pi*np.sqrt(5.4*10e-2*config.A_1)
    return h

def setCircularField(N,N_steps, dir = 0, hargs = ()):
    """Calculates the circular magnetic field at all times t based on the formula
    B_0 hatz (cos (omega t) + sin(omega t + phi(90 degrees or 0)))*stepfunction(t2-t1)

    Args:
        N ([int]): [Number of spins]
        N_steps ([int]): [Number og discretized timesteps]
        dir (int, optional): [The direction of the rotating magnetic field]. Defaults to 0.
        hargs (tuple, optional): (B0,omega,phi,Nt1,Nt2,dt) are the parameters of the magnetic field

    Returns:
        [ndarray(N,Nt,3)]: [The magnetic for all the particles at all times for each direction]
    """
    B0, omega, phi, Nt1, Nt2, dt = hargs
    h = np.zeros((N,N_steps,3), dtype = config.dtype)
    t = np.arange(Nt1,Nt2,1)*dt
    h[:,Nt1:Nt2,dir] = B0*(np.cos(omega*t) + np.sin(omega*t + phi))
    
    return h

#DIFFERENT ANISOTROPY INITIALIZERS

#Setting anisotropy using different initialization functions. 
#The general anisotropy matrix is given by a 3x3 matrix. 
#There is general flexibility in how the anisotropy of the system should be enforced. One may
#have only diagonal elements along this anisotropy tensor to indicate soft or hard axes, or 
#one may also have off-diagonal anisotropies if that is needed. 
#Soft and hard anisotropies are given by negative and positive values respectively.
#Alternatively, it is easier to simply initialize the anistropy using the main.py function 
# instead by first initializing the anisotropy matrix as a zeros(3,3) matrix and manually entering the values. 
def setAnisotropy(direction, value = -5.4*10e-2):
    """Use this funciton to set an anisotropy in a certain direction, It essentially fills in elements of an anisotropy tensor
    Note that if one calls this function twice for different direction, it is possible to set two uniaxial anisotropies,
    which creates a hard-axis and an easy exis. 

    Args:
        direction (string): a string with possible values 'x','y','z' depending on which direction you'd like the anisotropy in
        value (float, optional):  The value of the anisotropy. Defaults to -5.4*10e-2.
    """
    if direction == "x" or direction == "X":
        config.w_1[0,0] = value
    elif direction == "y" or direction == "Y":
        config.w_1[1,1] = value
    elif direction == "z" or direction == "Z":
        config.w_1[2,2] = value
    
def setAnisotropyMatrix(w_1, w_2 = np.zeros((3,3))):
    """Manually set the anisotropy tensors w1 and w2

    Args:
        w_1 (np.ndarray(3,3)): A 3x3 array which is the anisotropy-matrix.
        w_2 (np.ndarray(3,3), optional): A 3x3 array which sets the anisotropy-matrix. Defaults to np.zeros((3,3)).
    """
    config.w_2 = w_2
    config.w_1 = w_1

#DIFFERENT SPIN-SYSTEM INITIALIZATIONS

#One may also create their own way to initialize the spin lattice at the beginning of the simulation. 
#The only requirement for it to work is that the initital spin-lattice being initialized is of shape
# (Nx,Ny,Nz,N_steps,3) for the 3D case 
# (Nx,Ny,N_steps,3) for the 2D case
# (Nx,N_steps,3) for the 1D case
#For instance, if one were to initialize a 1D system according to some custom routine, one would
#first initialize the matrix, then fill inn the elements of the matrix m_a[:,0,:] since this 
#indicates the initial state of the m_a spins for all lattice positions on sublattice A. Similar
#procedure must be performed on the m_b to initalize it. Remember also to normalize the spins.
#Examples of these types of initalizations are provided in the following functions. 

def initInGivenDirection(N, N_steps,border_size,Ny = 0, Nz = 0,initargs = ()):
    """This function initalizes the m_a sublattice spins in the direction given by initargs, and then puts some random fluctiations on top.
    m_b is initialized in the oppositite direction of m_a with some random fluctuations on top.

    Args:
        N (int): Number of lattice sites.
        N_steps (int): Number of time-steps
        initargs (tuple(vector(3))): Takes in a tuple with one element which is a three vector with the direction which we wish to
        initialize the m_a spin in. ([x,y,z]) is how one inputs this function. 

    Returns:
        m_a (np.ndarray(N,N_steps,3)): Returns the m_a sublattice spins, of size (N,N_steps,3) initialized as described above
        m_a (np.ndarray(N,N_steps,3)): Returns the m_b sublattice spins, of size (N,N_steps,3) initialized as described above
    """
    if Ny == 0 and Nz == 0:
        m_a = np.zeros((N,N_steps,3))
        m_b = np.zeros((N,N_steps,3))
        start_vec = np.array(initargs)
        m_a[:,0,:] = start_vec
        m_b[:,0,:] = -1*start_vec
        m_a[border_size:N-border_size,0,:] += np.random.rand(N-2*border_size,3)*0.01 - 0.005
        m_b[border_size:N-border_size,0,:] += np.random.rand(N-2*border_size,3)*0.01 - 0.005
        normalize(m_a[:,0,:],m_b[:,0,:])
    elif Ny != 0 and Nz == 0:
        m_a = np.zeros((N,Ny,N_steps,3))
        m_b = np.zeros((N,Ny,N_steps,3))
        start_vec = np.array(initargs)
        m_a[:,:,0,:] = start_vec
        m_b[:,:,0,:] = -1*start_vec
        
        m_a[border_size:N-border_size,border_size:Ny-border_size,0] += np.random.rand(N-2*border_size,Ny-2*border_size,3)*0.01 - 0.005
        m_b[border_size:N-border_size,border_size:Ny-border_size,0] += np.random.rand(N-2*border_size,Ny-2*border_size,3)*0.01 - 0.005
        normalize(m_a[:,:,0,:],m_b[:,:,0,:])
    return np.array([m_a,m_b])

def initExpSetup(Nx,Ny,timesize):
    m_a = np.zeros((Nx,Ny,timesize,3))
    m_b = np.zeros((Nx,Ny,timesize,3))
    m_a[:,:50, 0, :] = np.array([1,0,0])
    m_a[:,150:250, 0, :] = np.array([1,0,0])
    m_a[:,250:350, 0, :] = np.array([1,0,0])
    m_b[:,:50, 0, :] = -1*np.array([1,0,0])
    m_b[:,150:250, 0, :] = -1*np.array([1,0,0])
    m_b[:,250:350, 0, :] = -1*np.array([1,0,0])
    m_a[:,350:,0,:] = np.array([1,0,0])
    m_b[:,350:,0,:] = -1*np.array([1,0,0])
    
    
    m_a[:,50:150, 0, :] = np.array([0,1,0])
    m_b[:,50:150, 0, :] = -1*np.array([0,1,0])
    m_a[:,250:350, 0, :] = np.array([0,1,0])
    m_b[:,250:350, 0, :] = -1*np.array([0,1,0])
    
    m_a[:,:,0] += np.random.rand(Nx,Ny,3)*0.01 - 0.005
    m_b[:,:,0] += np.random.rand(Nx,Ny,3)*0.01 - 0.005
    normalize(m_a[:,:,0,:],m_b[:,:,0,:])
    return np.array([m_a,m_b])



def initGSFromtxt(N,N_steps, matxt = "GS/x-dir/m_a_gs_x.txt", mbtxt = "GS/x-dir/m_b_gs_x.txt", *initargs):
    """Initializes a 1D system according to the contents of a .txt file. 

    Args:
        N ([int]): Number of lattice sites
        N_steps ([int]): Number of timesteps
        matxt ([string]): String of the .txt file that contains the initial state of the m_a spins
        mbtxt ([string]): String of the .txt file that contains the initial state of the m_b spins
    Returns:
        [np.ndarray(N,N_steps,3)]: The m_a and m_b arrays are returned
    """
    m_a = np.zeros((N,N_steps,3))
    m_b = np.zeros((N,N_steps,3))
    dir = initargs[0]
    m_atxt = np.loadtxt(matxt)
    m_btxt = np.loadtxt(mbtxt)
    m_a[:,dir,:] = m_atxt[:N,:]
    m_b[:,dir,:] = m_btxt[:N,:]
    return m_a, m_b

def normalize(m_1, m_2):
    """A call to this function will normalize the vectors. 

    Args:
        m_1 (np.ndarray(N,3)): A time-instance of the total spin matrix. It is of dimension (N,3)
        m_2 (np.ndarray(N,3)): A time-instance of the other sublattice spins. It is of timension (N,3)
    """
    if len(np.shape(m_1)) == 2:
        norms = np.linalg.norm(m_1[:,:], axis = 1)
        m_1[:, 0] = m_1[:, 0] / norms[:]
        m_1[:, 1] = m_1[:, 1] / norms[:]
        m_1[:, 2] = m_1[:, 2] / norms[:]

        norms = np.linalg.norm(m_2[:,:], axis = 1)
        m_2[:, 0] = m_2[:, 0] / norms[:]
        m_2[:, 1] = m_2[:, 1] / norms[:]
        m_2[:, 2] = m_2[:, 2] / norms[:]
    else:
        norms = np.linalg.norm(m_1[:,:,:], axis = 2)
        m_1[:,:, 0] = m_1[:,:, 0] / norms[:,:]
        m_1[:,:, 1] = m_1[:,:, 1] / norms[:,:]
        m_1[:,:, 2] = m_1[:,:, 2] / norms[:,:]

        norms = np.linalg.norm(m_2[:,:], axis = 2)
        m_2[:,:, 0] = m_2[:,:, 0] / norms[:,:]
        m_2[:,:, 1] = m_2[:,:, 1] / norms[:,:]
        m_2[:,:, 2] = m_2[:,:, 2] / norms[:,:]
  
def initWithFunction(N,N_steps, *initargs):
    """Initialize the x,y,z components of the spins using a potentially custom function. The arguments of these functions are 
    np.linspace(Lmin,Lmax,N), it the fucton functions needs to be able to evaluate something of that form. some examples of these
    functions are np.tanh, np.cos, np.sin or some other custom function such as the defaultFunction.

    Args:
        N (int): Number of lattice sites.
        N_steps (int): Number of time-steps
        initargs:
            Lmin: The leftmost edge of the function evalutations
            Lmax: The rightmost edge of the function evalutations
            function1: The function that initializes the x-components of the spins. 
            function2: The function that initializes the y-components of the spins. 
            function3: The function that initializes the z-components of the spins. 
    Returns:
        m_a (np.ndarray(N,N_steps,3)): Returns the m_a sublattice spins, of size (N,N_steps,3) initialized as described above
        m_a (np.ndarray(N,N_steps,3)): Returns the m_b sublattice spins, of size (N,N_steps,3) initialized as described above
    """
    m_a = np.zeros((N,N_steps,3)) 
    Lmin,Lmax,function1,function2,function3 = initargs
    m_a[:,0,0] = function1(np.linspace(Lmin,Lmax,N))
    m_a[:,0,1] = function2(np.linspace(Lmin,Lmax,N))
    m_a[:,0,2] = function3(np.linspace(Lmin,Lmax,N))
    m_b = -m_a
    normalize(m_a[:,0,:],m_b[:,0,:])
    return m_a,m_b

def randomizeStart(N,N_steps, Ny = 0, Nz = 0,border_size = 0, *initargs):
    """Uses the randomizeVectors subroutine to create an initial state of vectors that point in random directions
    It then returns the complete spin-matrices of dimension (N,N_steps,3) so that the dynamics can happen. 

    Args:
        N (int): Number of vectors on the lattice
        N_steps (int): Number of timesteps that are going to be simulated. 
        initargs: There are no neccessary initargs in this initializer routine. 
    Returns:
        m_a_start: Returns the total m_a spin-matrix of dimension (N,N_steps,3) of randomized vectors
        m_b_start: Returns the total m_b spin-matrix of dimension (N,N_steps,3) of randomized vectors 
    """
    if Ny == 0 and Nz == 0:
        m_a_start = np.zeros((N,N_steps,3))
        m_b_start = np.zeros((N,N_steps,3))
        m_a_start[N-config.border_size-1 : N,:, 0] = 1
        m_b_start[N-config.border_size-1 : N,:,0] = -1
        m_a_start[0:config.border_size,:,0] = 1
        m_b_start[0:config.border_size,:,0] = -1
        m_a_start[:, 0, :], m_b_start[:, 0, :] = randomizeVectors(m_a_start[:, 0, :], m_b_start[:, 0, :])
    else:
        m_a_start = np.zeros((N,Ny,N_steps,3))
        m_b_start = np.zeros((N,Ny,N_steps,3))
        m_a_start[N-config.border_size-1 : N,:,:, 0] = 1
        m_b_start[N-config.border_size-1 : N,:,:,0] = -1
        m_a_start[0:config.border_size,:,:,0] = 1
        m_b_start[0:config.border_size,:,:,0] = -1

        m_a_start[:,Ny-config.border_size-1 : Ny,:, 0] = 1
        m_b_start[:,Ny-config.border_size-1 : Ny,:, 0] = -1
        m_a_start[:,:config.border_size,:, 0] = 1
        m_b_start[:,:config.border_size,:,0] = -1
        m_a_start[:,:, 0, :], m_b_start[:,:, 0, :] = randomizeVectors2D(m_a_start[:,:, 0, :], m_b_start[:,:, 0, :])
    return np.array([m_a_start, m_b_start])

def randomizeVectors2D(m_1,m_2):
    """Creates randomized vectors that initialize the system. 

    Args:
        m_1 (np.ndarray(Nx,Ny,3)): The first time-element of the m_a matrix. 
        m_2 (np.ndarray(Nx,Ny,3)): The first time-element of the m_b matrix. 

    Returns:
        m_1: Returns randomized vectors 3-vectors on each lattice site 
        m_2: Returns randomized vectors 3-vectors on each lattice site. 
    """
    N = m_1.shape[0]
    Ny = m_1.shape[1]
    m_1[config.border_size:N-config.border_size,config.border_size:Ny-config.border_size,:] = np.random.rand(N-2*config.border_size,Ny - 2*config.border_size,3) - 0.5
    norms = np.linalg.norm(m_1[:,:,:], axis = 2)
    m_1[:,:,0] = m_1[:,:, 0] / norms[:,:]
    m_1[:,:,1] = m_1[:,:, 1] / norms[:,:]
    m_1[:,:,2] = m_1[:,:, 2] / norms[:,:]
    m_2[config.border_size:N-config.border_size,config.border_size:Ny-config.border_size,:] = np.random.rand(N-2*config.border_size,Ny - 2*config.border_size,3) - 0.5
    norms = np.linalg.norm(m_2[:,:,:], axis = 2)
    m_2[:,:,0] = m_2[:,:, 0] / norms[:,:]
    m_2[:,:,1] = m_2[:,:, 1] / norms[:,:]
    m_2[:,:,2] = m_2[:,:, 2] / norms[:,:]
    return m_1, m_2

def randomizeVectors(m_1, m_2):
    """Creates randomized vectors that initialize the system. 

    Args:
        m_1 (np.ndarray(N,3)): The first time-element of the m_a matrix. 
        m_2 (np.ndarray(N,3)): The first time-element of the m_b matrix. 

    Returns:
        m_1: Returns randomized vectors 3-vectors on each lattice site 
        m_2: Returns randomized vectors 3-vectors on each lattice site. 
    """
    N = m_1.shape[0]
    m_1[config.border_size:N-config.border_size, :] = np.random.rand(N-2*config.border_size,3) - 0.5
    norms = np.linalg.norm(m_1[config.border_size:N-config.border_size, :], axis = 1)
    m_1[config.border_size:N-config.border_size, 0] = m_1[config.border_size:N-config.border_size, 0] / norms[:]
    m_1[config.border_size:N-config.border_size, 1] = m_1[config.border_size:N-config.border_size, 1] / norms[:]
    m_1[config.border_size:N-config.border_size, 2] = m_1[config.border_size:N-config.border_size, 2] / norms[:]
    m_2[config.border_size:N-config.border_size, :] = np.random.rand(N- 2*config.border_size,3) - 0.5
    norms = np.linalg.norm(m_2[config.border_size:N-config.border_size, :], axis = 1)
    m_2[config.border_size:N-config.border_size, 0] = m_2[config.border_size:N-config.border_size, 0] / norms[:]
    m_2[config.border_size:N-config.border_size, 1] = m_2[config.border_size:N-config.border_size, 1] / norms[:]
    m_2[config.border_size:N-config.border_size, 2] = m_2[config.border_size:N-config.border_size, 2] / norms[:]
    return m_1, m_2

#DIFFERENT TORQUE TERMS (INPUT CURRENT TERMS)

#For a custom generated input current, one has to return a numpy array of shape 
# (Nx,Ny,Nz,N_steps,3) for the 3D case 
# (Nx,Ny,N_steps,3) for the 2D case
# (Nx,N_steps,3) for the 1D case
#The way to initialize the torque term takes the same form as the initialization of the magnetic field.
#One may configure the torque term at all times and individually target different regions of space
#as one pleases using a custom routine.

def constantTorque(N,N_steps, dir = 0, value = 1):
    """Creates a constant torque term for a 1D system

    Args:
        N ([int]): Size of the 1D system
        N_steps ([int]): Number of timesteps of the simulation
        dir (int, optional): direction of the torque term. 0 = x, 1 = y, 2 = z. Defaults to 0.
        value (int, optional): Strength of the toque term. Defaults to 1.

    Returns:
        [np.ndarray(N,N_steps,3)]: Returns the toeque for all spins at all lattice positions at all times. 
    """
    tau = np.zeros((N,N_steps,3), dtype = config.dtype)
    tau[:20,15000:,dir] = value 
    return tau

def setInputCurrent(Nx,N_steps, current_length,od_length,direction_vector = np.array([0,0,1]),on = True, from_step = 0, to_step = 0, Ny = 0):
    """Generates the Input current for all times t as an input to the function. 

    Args:
        Nx (int): Number of spins in the x direction
        N_steps (int): Number of time-steps
        current_length (int): how far into the material the current is inputted
        direction_vector (np.ndarray(3), optional): the direction of the arrow. Does not need to be normalized. Defaults to np.array([0,0,1]).
        on (bool, optional): If true, the input current is turned on from time 0 to timestep 5000. Defaults to False.

    Returns:
        [np.ndarray(N,N_steps,3)]: returns the input current-matrix for all times. If the system is 2D, it returns a matrix of size (Nx,Ny,N_steps,3)
    """
    if Ny == 0:
        J = np.zeros((Nx,N_steps,3))
        direction_vector = direction_vector/(np.sqrt(np.sum(direction_vector**2)))
        if on:
            J[od_length:(current_length+od_length),from_step:to_step,:] = direction_vector
    else:
        J = np.zeros((Nx,Ny,N_steps,3))
        direction_vector = direction_vector/(np.sqrt(np.sum(direction_vector**2)))
        if on:
            J[od_length:(current_length+od_length),:,from_step:to_step,:] = direction_vector
    return J

def setAlpha(Nx,current_length,alpha_g,alpha_sp,alpha_overdamping, od_length = 5, Ny = 0):
    """Generates the damping matrix, since it is spatially dependent, with overdamping on the edges, alpha_sp due to
    the input current and output current

    Args:
        Nx (int): Number of lattice points
        current_length (int): how far into the material the current 
        alpha_g (float): gilbert damping
        alpha_sp (float): damping due to currents
        alpha_overdamping (float): the overdamping alpha

    Returns:
        alpha (np.ndarray(N,3)): Returns the spatially dependent damping matrix. 
    """
    if Ny == 0:
        alpha = alpha_g*np.ones((Nx,3))
        alpha[:od_length,:] = alpha_overdamping
        alpha[od_length:current_length + od_length,:] += alpha_sp
        alpha[Nx-od_length:,:] = alpha_overdamping
        alpha[Nx-current_length-od_length:Nx-od_length,:] += alpha_sp
        config.od_length = od_length
        config.alph = alpha
        config.current_length = current_length
    else:
        alpha = alpha_g*np.ones((Nx,Ny,3))
        alpha[:od_length,:,:] = alpha_overdamping
        alpha[:,:od_length,:] = alpha_overdamping
        alpha[od_length:current_length + od_length, :, :] += alpha_sp
        alpha[Nx-od_length-1:,:] = alpha_overdamping
        alpha[:,Ny-od_length-1:,:] = alpha_overdamping
        alpha[Nx-current_length-od_length-1:Nx-od_length,:,:] += alpha_sp
        config.od_length = od_length
        config.alph = alpha
        config.current_length = current_length
    return alpha

#SETTING TEMPRERATURE

#Setting the temperature is similar to the way one sets the damping term. The returned shape of 
#a custom function should be of the shape mentioned earlier, notably 1D: (Nx,3), 2D: (Nx,Ny,3)

def initConstTemp(Nx,Ny = 0, amplitude = 0.0):
    """Initialize a constant temperature system with a given amplitude. 

    Args:
        Nx ([int]): Number of lattice positions in x direction
        Ny (int, optional): Number of lattice positions in the y direction. Defaults to 0.
        amplitude (float, optional): [description]. Defaults to 0.0.

    Returns:
        [np.ndarray(Nx,3)]: Returns the temperature at all spin positions. If a 2D system, it returns np.ndarray(Nx,Ny,3)
    """
    if Ny == 0:
        return np.ones((Nx,3))*amplitude
    else:
        return np.ones((Nx,Ny,3))*amplitude

def CheckRelaxed(ma,mb,tol = 1e-5,checkpoint = 1000):
    """Checks wether the a 1D system is relaxed by seeing wether the change in spin-behaviour after checkpoint number of timesteps is less than a given tolerance

    Args:
        ma ([np.ndarray(Nx,Nsteps,3)]): The m_a spin matrix
        mb ([np.ndarray(Nx,Nsteps,3)]): The m_b spin matrix
        tol ([float], optional): The allowed tolerance . Defaults to 1e-5.
        checkpoint (int, optional): Compare the last state to the state checkpoint timesteps prior to the last step. Defaults to 1000.

    Returns:
        [bool]: Returns true if the change is smaller than the tolerance, false if that is not the case
    """
    if np.linalg.norm(ma[:,-1,:] - ma[:,-checkpoint,:]) > tol:
        return False
    return True

def SItoRuntimeConstants(constants, M_s):
    """Transfers SI constants to runtime constants ATTENTION::NOT FOR J AND H, these are handled manually

    Args:
        constants([tuple]) : Array of constants in SI, see main
        M_s([float]) : Saturation magnetization
    Returns:
        constants([tuple]) : Array of constants scaled to runtime units, see main
    """
    w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J_func, J_args,d,h_func, h_args,D, Temp = constants
    return (w_ex/M_s,A_1/M_s,A_2/M_s,w_1/M_s,w_2/M_s,beta/M_s,C/M_s,alpha,J_func, J_args,d/M_s,h_func, h_args,D/M_s, Temp/M_s)

def runtimeToSIConstants(constants, M_s):
    """Transfers SI constants to runtime constants ATTENTION::NOT FOR J AND H, these are handled manually

    Args:
        constants([tuple]) : Array of constants in Runtime units, see main
        M_s([float]) : Saturation magnetization
    Returns:
        constants([tuple]) : Array of constants scaled to SI units, see main
    """
    w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J_func, J_args,d,h_func, h_args,D, Temp = constants
    return (w_ex*M_s,A_1*M_s,A_2*M_s,w_1*M_s,w_2*M_s,beta*M_s,C*M_s,alpha,J_func, J_args,d*M_s,h_func, h_args,D*M_s, Temp*M_s)


if __name__ == '__main__':
    print(0)
    #currrentThresholdAnalyser(np.linspace(1,5,10), np.linspace(0.01,0.1, 10))
    #tempScan()
    #hscanSpinFlop()