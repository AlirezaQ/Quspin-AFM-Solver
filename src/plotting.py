import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import config
plt.switch_backend('Qt5Agg')


@nb.jit(nopython = True, fastmath= True)
def crossProduct(v1,v2):
    """Code to do a cross product, need it to be able to njit
    Args:
        v1 (ndarray): first vector 
        v2 (ndarray): second vector
    """
    res = np.zeros_like(v1)
    res[:,0] = v1[:,1]*v2[:,2] - v1[:,2]*v2[:,1]
    res[:,1] = v1[:,2]*v2[:,0] - v1[:,0]*v2[:,2]
    res[:,2] = v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
    return res

def SpinAnimationWrapper(spins_1, spins_2, steps_along_axis = 4,average = False):
    """A wrapper function for the animation-procedure. For the animation to not include too many frames,
    the input-parameter steps takes care of making the animation go a bit faster by jumping over some
    time-steps. This animation plots both the m_a and m_b spins. 

    Args:
        spins_1 ([np.ndarray(Nx,Ny,Nsteps,3)]): [description]
        spins_2 ([np.ndarray(Nx,Ny,Nsteps,3)]): [description]
        Nx ([int]): Number of lattice points along x direction
        N_steps ([int]): [Number of timestep total in the simulation]
        stride ([int]): [The stride'th timestep is saved in the overarching m_a,m_b array]
        Ny (int, optional): Number of lattice points along y direction. This is 0 if there is a 1D system
        steps_along_axis (int, optional): This integer skips steps_along_axis number of spins when plotting. Default to 1
    """
    shape_tup = spins_1.shape
    Nx = shape_tup[0]
    if len(shape_tup) == 4:
        Ny = shape_tup[1]
        N_steps = shape_tup[2]
    else:
        N_steps = shape_tup[1]
    if Ny ==0:
        if average == True:
            averageMa = np.mean(np.array(np.split(spins_1,int(Nx/steps_along_axis),axis = 0)),axis = 1)
            averageMb = np.mean(np.array(np.split(spins_2,int(Nx/steps_along_axis),axis = 0)),axis = 1)
            Animate1D(int(Nx/steps_along_axis),(N_steps), averageMa ,averageMb)
        else:
            Animate1D(int(Nx/steps_along_axis),(N_steps), spins_1[::steps_along_axis] ,spins_2[::steps_along_axis])
    elif Ny!= 0:
        if average == True:
            averageMa = np.average(np.array(np.split(np.array(np.split(spins_1,int(Nx/steps_along_axis),axis = 0)),int(Ny/steps_along_axis),2)),axis = (2,3))
            averageMb = np.average(np.array(np.split(np.array(np.split(spins_2,int(Nx/steps_along_axis),axis = 0)),int(Ny/steps_along_axis),2)),axis = (2,3))
            Animate2D(int(Nx/steps_along_axis),int(Ny/steps_along_axis), (N_steps), averageMa, averageMb)
        else:
            Animate2D(int(Nx/steps_along_axis),int(Ny/steps_along_axis), (N_steps), spins_1[::steps_along_axis,::steps_along_axis], spins_2[::steps_along_axis,::steps_along_axis])
    
def Animate1D(N: int, N_steps: int, spins_1: np.ndarray, spins_2:np.ndarray):
    """This is a function which performs a slider-animation of the spins in 1D.

    Args:
        N (int): Number of spins on the lattice
        N_steps (int): The number of time-steps that are to be plotted. The number of steps is post-slicing of the lists. 
        spins_1 (np.ndarray): The m_a spin-matrix of size (N,N_steps,3)
        spins_2 (np.ndarray): The m_b spin-matrix of size (N,N_steps,3)

    Returns:
        [None]: This function performs the animation, so it does not return anything. 
    """
    global quiver
    global quiver2
    global s_it2
    global fig2
    global ax2
    global c
    c = "r"

    def get_arrow(frame:int, spins: np.ndarray,N:int)-> tuple:
        """This is the function that is called as a frame in the animation takes place.
        It extracts the frame'th iteration of the spins-matrix and uses it to generate the plot
        The funcion first generates a meshgrid to create to locations of the spins, x,y,z
        and then extracts the spin-data in order to get the arrow-location u,v,w which are the
        x,y,z components of the spin respectively.

        Args:
            frame (int): The frame'th component of the velocity component. 
            spins (np.ndarray): [description]
            N (int): Number of spns on the lattice.

        Returns:
            tuple: A tuple which consists of (x,y,z,u,v,w) where x,y,z are a meshgrid of 
            lattice-positions, while u,v,w are the velocities of the spins. 
        """
        x, y, z = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, 1), np.linspace(0, 1, 1))

        u = np.zeros((1, N, 1))
        v = np.zeros((1, N, 1))
        w = np.zeros((1, N, 1))

        u[0, :, 0] = spins[:,int(frame),0]
        v[0, :, 0] = spins[:,int(frame),1]
        w[0, :, 0] = spins[:,int(frame),2]
        return x, y, z, u, v, w

    def update_quiver(frame: int) -> None:
        """
        This is the update function which is called every time an iteratin of slider-animation happens.
        It first removes the canvas, and then plots the spins usin the get_arrow function. It plots both
        m_a and m_b
        """
        global quiver
        global quiver2
        global ax2
        frame = s_it2.val
        quiver.remove()
        quiver2.remove()
        quiver = ax2.quiver(*get_arrow(frame, spins_1,N),length = 0.5,color = "b")
        quiver2 = ax2.quiver(*get_arrow(frame, spins_2, N), length = 0.5, color = "r")

    #fig2, ax2 = plt.subplots(subplot_kw=dict(projection="3d"))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection = "3d")
    
    quiver = ax2.quiver(*get_arrow(0, spins_1,N), length = 0.5, color = "b", label = r'$m_a$')
    quiver2 = ax2.quiver(*get_arrow(0,spins_2,N), length = 0.5, color = "r", label = r'$m_b$')
    plt.legend()
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-1.5, 1.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    s_it2 = Slider(fig2.add_subplot(50,1,50),  valmin = 0, valmax = N_steps-1, valinit = 0, label = "Iterations")
    s_it2.on_changed(update_quiver)
    plt.show()

def Animate2D(N: int,Ny:int, N_steps: int, spins_1: np.ndarray, spins_2:np.ndarray):
    """This is a function which performs a slider-animation of the spins using arrows in each lattice position in 2D.

    Args:
        N (int): Number of spins on the lattice
        N_steps (int): The number of time-steps that are to be plotted. The number of steps is post-slicing of the lists. 
        spins_1 (np.ndarray): The m_a spin-matrix of size (N,N_steps,3)
        spins_2 (np.ndarray): The m_b spin-matrix of size (N,N_steps,3)

    Returns:
        [None]: This function performs the animation, so it does not return anything. 
    """
    global quiver
    global quiver2
    global s_it2
    global fig2
    global ax2
    global c
    c = "r"

    def get_arrow(frame:int, spins: np.ndarray,N:int, Ny: int)-> tuple:
        """This is the function that is called as a frame in the animation takes place.
        It extracts the frame'th iteration of the spins-matrix and uses it to generate the plot
        The funcion first generates a meshgrid to create to locations of the spins, x,y,z
        and then extracts the spin-data in order to get the arrow-location u,v,w which are the
        x,y,z components of the spin respectively.

        Args:
            frame (int): The frame'th component of the velocity component. 
            spins (np.ndarray): [description]
            N (int): Number of spns on the lattice.

        Returns:
            tuple: A tuple which consists of (x,y,z,u,v,w) where x,y,z are a meshgrid of 
            lattice-positions, while u,v,w are the velocities of the spins. 
        """
        x, y, z = np.meshgrid(np.linspace(0, 1, spins.shape[0]), np.linspace(0, 1, spins.shape[1]), np.linspace(0, 1, 1))
        u = np.zeros((spins.shape[1], spins.shape[0],1))
        v = np.zeros_like(u)
        w = np.zeros_like(u)
        u[:, :, 0] = spins[:,:,int(frame),0].T
        v[:, :, 0] = spins[:,:,int(frame),1].T
        w[:, :, 0] = spins[:,:,int(frame),2].T
        return x, y, z, u, v, w

    def update_quiver(frame: int) -> None:
        """
        This is the update function which is called every time an iteratin of slider-animation happens.
        It first removes the canvas, and then plots the spins usin the get_arrow function. It plots both
        m_a and m_b
        """
        global quiver
        global quiver2
        global ax2
        frame = s_it2.val
        quiver.remove()
        quiver2.remove()
        quiver = ax2.quiver(*get_arrow(frame, spins_1,N, Ny),length = 0.1, linewidths = 0.6, color = "b")
        quiver2 = ax2.quiver(*get_arrow(frame, spins_2, N,Ny), length = 0.1, linewidths = 0.6, color = "r")

    #fig2, ax2 = plt.subplots(subplot_kw=dict(projection="3d"))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111,projection="3d")
    
    quiver = ax2.quiver(*get_arrow(0, spins_1,N,Ny), length = 0.1,linewidths = 0.6, color = "b", label = '$m_a$')
    quiver2 = ax2.quiver(*get_arrow(0,spins_2,N,Ny), length = 0.1,linewidths = 0.6, color = "r", label = '$m_b$')
    plt.legend()
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_zlim(-1.5, 1.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    s_it2 = Slider(fig2.add_subplot(50,1,50),  valmin = 0, valmax = N_steps-1, valinit = 0, label = "Iterations")
    s_it2.on_changed(update_quiver)
    plt.show()

def grapher(m_a: np.ndarray,m_b:np.ndarray,ylimit:float = 1.5,type_of_plot:str = 'm'):
    """This procedure does the graphing of the x,y,z components of both the m_a and m_b sublattices.

    Args:
        m_a (np.ndarray): m_a spin-matrix of size (N,N_steps,3)
        m_b (np.ndarray): m_b spin-matrix of size (N,N_steps,3)
        ylimit (float): Sets the limit of the y-axis of the subplots. Defaults to 1.5 for plotting the component of the spin. 
        type_of_plot (str): A string that is used to set the title of the subplots. 
    """
    #There is a lot of globals here, since the update function needs to know about them
    global fig
    global ls
    global s_it
    N = np.shape(m_a)[0]
    N_steps = np.shape(m_a)[1] 
    X_arr = np.arange(0,N)*config.delta*10**7 #(nm)
    fig = plt.figure()
    #fig.canvas.set_window_title('spin simulations after %d steps' %(N_steps))
    ls = []
    frame = 10

    #Making plots for 3d along x
    ax = fig.add_subplot(3,3,1)
    #ax.set_ylim(-ylimit, ylimit)
    l, = ax.plot(X_arr, m_a[:,0,0])
    ls.append(l)
    ax.set_title(f'${type_of_plot}_a x$')
    ax.set_ylim(-ylimit, ylimit)
    ax = fig.add_subplot(3,3,2)
    l, = ax.plot(X_arr, m_a[:,0,1])
    ls.append(l)
    ax.set_title(f'${type_of_plot}_a y$')
    ax.set_ylim(-ylimit, ylimit)
    ax = fig.add_subplot(3,3,3)
    l, = ax.plot(X_arr, m_a[:,0,2])
    ls.append(l)
    ax.set_title(f'${type_of_plot}_a z$')
    ax.set_ylim(-ylimit, ylimit)
    ax = fig.add_subplot(3,3,4)
    l, = ax.plot(X_arr, m_b[:,0,0])
    ls.append(l)
    ax.set_title(f'${type_of_plot}_b x$')
    ax.set_ylim(-ylimit, ylimit)
    ax = fig.add_subplot(3,3,5)
    l, = ax.plot(X_arr, m_b[:,0,1])
    ls.append(l)
    ax.set_title(f'${type_of_plot}_b y$')
    ax.set_ylim(-ylimit, ylimit)
    ax = fig.add_subplot(3,3,6)
    l, = ax.plot(X_arr, m_b[:,0,2])
    ls.append(l)
    ax.set_title(f'${type_of_plot}_b z$')
    ax.set_ylim(-ylimit, ylimit)
    s_it = Slider(fig.add_subplot(50,1,50),  valmin = 0, valmax = N_steps-1, valinit = 0, label = "Iterations")
    
    #Function that updates values on changing the slider value
    def update(val):
        frame = s_it.val
        ls[0].set_data(X_arr, m_a[:,int(frame),0])
        ls[1].set_data(X_arr, m_a[:,int(frame),1])
        ls[2].set_data(X_arr, m_a[:,int(frame),2])
        ls[3].set_data(X_arr, m_b[:,int(frame),0])
        ls[4].set_data(X_arr, m_b[:,int(frame),1])
        ls[5].set_data(X_arr, m_b[:,int(frame),2])

    s_it.on_changed(update)
    plt.show()

def JOut1D(mdot_a, mdot_b, m_a, m_b, dt, T, beta, ol, cl):
    """Function to calculate and plot the spin current at the end of system in 1D

    Args:
        mdot_a (ndarray): Time-derivative of m_a
        mdot_b (ndarray): Time-derivative of m_b
        m_a (ndarray): Magnetisation in sublattice A
        m_b (ndarray): Magnetisation in sublattice B
        dt (float): timestep
        T (float): Total dim-less time (N_steps*dt)
        beta (float): Spin current strength (see main)
        ol (int): overdamping-length, the size of the overdamping at the end of system in lattice points
        cl (int): current-length, the size of the current region 
    """
    #T = np.arange()
    #Partition mdot and m into only the lead
    calc_ma, calc_mb = m_a[m_a.shape[0]-cl -ol:m_a.shape[0]-ol, :, :], m_b[m_a.shape[0]-cl -ol:m_a.shape[0]-ol, :, :]
    calc_mdot_a, calc_mdot_b =  mdot_a[m_a.shape[0]-cl -ol:m_a.shape[0]-ol, :, :], mdot_b[m_a.shape[0]-cl -ol:m_a.shape[0]-ol, :, :]
    calc_ma_in, calc_mb_in = m_a[cl:cl +ol, :, :], m_b[cl:cl+ol, :, :]
    calc_mdot_a_in, calc_mdot_b_in =  mdot_a[cl:cl+ol, :, :], mdot_b[cl:cl +ol, :, :]
    #Calculate Js
    J_a,J_b = np.zeros_like(calc_ma),np.zeros_like(calc_mb)
    J_in_a,J_in_b = np.zeros_like(calc_ma_in),np.zeros_like(calc_mb_in)
    for i in range(J_a.shape[1]):
        J_a[:,i,:] = config.alpha_sp*crossProduct(calc_ma[:,i,:], calc_mdot_a[:,i,:])
        J_b[:,i,:] = config.alpha_sp*crossProduct(calc_mb[:,i,:], calc_mdot_b[:,i,:])
        J_in_a[:,i,:] = config.alpha_sp*crossProduct(calc_ma_in[:,i,:], calc_mdot_a_in[:,i,:])
        J_in_b[:,i,:] = config.alpha_sp*crossProduct(calc_mb_in[:,i,:], calc_mdot_b_in[:,i,:])
    J_in_z = 0.5*(np.mean(J_in_a[:, :, 2], axis = 0) + np.mean(J_in_b[:, :, 2], axis = 0))
    J_out_z = -0.5*(np.mean(J_a[:, :, 2], axis = 0) + np.mean(J_b[:, :, 2], axis = 0))
    print('het')
    J_out_prop_in = J_out_z/beta
    T = (T/config.f_0)*10**9
    np.savetxt("J_out.txt", J_out_prop_in)
    plt.figure()
    plt.xlabel("Time [ns]")
    plt.ylabel(r"$\frac{J_{in}}{J_{out}}$")
    plt.title("Current out")
    plt.plot(T,J_out_prop_in)
    plt.show()

def normSurfacePlot(m_a,m_b):
    """Helper plot to look at stability, plots the norm of m_a and m_b at the end of simulation in 2D
    
    Args:
        m_a (ndarray): magnetisation in sublattice A
        m_a (ndarray): magnetisation in sublattice B
    """
    matplotlib.get_backend()
    Nx = m_a.shape[0]
    Ny = m_a.shape[1]
    x, y = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))
    fig = plt.figure(figsize = (4,4))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    surf1 = ax1.plot_surface(x,y,np.linalg.norm(m_a[:,:, -1,:], axis = 2), cmap = "jet", antialiased = True)

    ax2 = fig.add_subplot(1,2,2, projection = '3d')
    surf2 = ax2.plot_surface(x,y,np.linalg.norm(m_b[:,:,-1,:], axis = 2), cmap = "jet", antialiased = True)
    plt.show()
        

def plotTemperatureScan():
    plt.figure()
    Temps = np.loadtxt(r"data\Temp-scan-easter\Temperatures.txt")
    Neel_avg = np.loadtxt(r"data\Temp-scan-easter\Neel_avg_1.txt")
    plt.plot(Temps,Neel_avg)
    plt.show()
if __name__ == '__main__':
    plotTemperatureScan()