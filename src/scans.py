import random
from sys import set_coroutine_origin_tracking_depth
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
import utilities as ut
plt.switch_backend('Qt5Agg')
#WHOLE PROCEDURES (EX. PLOTTING MEGNETIZATION AS A FUNCTION OF B-FIELD, ETC. )

#This section is dedicated to presenting whole simulations masked into functions. 
#These were used for a variety of purposes, however the most prominent
#ones are where we plotted the magnetization as a function of the external magnetic field
#in order to see the different magnetic phase transition of a 1D spin system.
#These functions give an idea as to how a potential run can be structured if one wishes to 
#perform custom routines for a variety of purposes. 

def hscanSpinFlop():
    """
    This is a whole routine for plotting the spin-flop transition.
    The essence of the simulation is to use different magnetic field strengths
    and run multiple simulations with each. Using the resulting m_a, m_b matrices
    one may calculate the magnetization at the end of each simulation. 
    From there remains a simple plotting procedure in order to show the
    results from the magnetic field scan. 
    """
    g_e = 2.0
    alph = 0.0068
    w_ex = 6.4*10**2
    mumag = 5.788e-5
    hbar = 6.582e-16
    D = 1
    d0 = 1
    dtype = np.float64
    gam = g_e * mumag / hbar
    w_ex2 = 1
    A = (1/(2*D))*w_ex
    A_1 = 0.5*w_ex/8 #w_ex*3/8#A
    A_2 = A_1 - A#A_1 -A
    w_1 = np.zeros((3,3), dtype = dtype)
    w_2 = np.zeros((3,3), dtype = dtype)
    w_h = np.sqrt(2*abs(w_ex)*2.4e-2)
    w_hc = np.sqrt(2.4*10**-2*(2*w_ex + 2.4*10**-2))
    border_size  = 0
    mu_0 = 4.0 * np.pi*10**-7
    M_s = 1.94e5
    f_0 = gam*mu_0*M_s
    d_0 = 4.17e-10
    t_0 = f_0**(-1)
    r_0 = np.sqrt(w_ex*d_0**2/(2*D))
    J = 0
    d = np.array([0,0,0])
    C = 0
    w_m = 4*np.pi*mu_0*M_s*gam/f_0
    current_length = 0
    od_length = 0
    alpha_sp = 0.02
    delta = 4*3.549*2*d_0
    flopH = np.sqrt(2.4*10**-2*(2*w_ex + 2.4*10**-2))
    max_h = flopH*2 #2*floph
    min_h = 0
    h_strengths = np.arange(min_h,max_h, 0.3)
    angles = np.zeros_like(h_strengths)
    #simulation constants
    L = 58
    dx = 1  
    N = int(L/dx)
    N_steps = 30000
    current_length = int(0.2*N)
    od_length = int(0.1*N)
    border_size = 0

    #Prepare physical constants
    alpha  = setAlpha(N,current_length, alph, alph, alph, od_length=od_length) #Damping
    J = setInputCurrent(N,N_steps, current_length,od_length,from_step = 0, to_step = N_steps, on = False) #Input current
    d = np.array([0,0,0]) #DMI vector
    w_1 = np.zeros((3,3))
    #positive is hard axis!!!
    
    #5.4e-1
    w = np.zeros_like(w_1)
    w_1[2,2] = 5.4e-1
    w_1[1,1] = -2.4e-2
    w_2 = -0.5*w_1
    beta = 0
    D = 0
    dt = 5*10**-3/config.w_m
    
    #Set inital state
    init_state = randomizeStart(N, N_steps)
    
    for i in range(len(h_strengths)):
        h = setHField(N,N_steps, hargs = (h_strengths[i]))
        #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d, D)
        constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, J, d, h, D)
        m_a, m_b, _, _, _ = RK4.runSimulation(N,N_steps, dt, init_state, constants = constants)    
        angles[i] = np.arccos(np.sqrt(np.mean(m_a[:,-1,1])**2)/(np.sqrt(np.mean(m_a[:,-1,0])**2 + np.mean(m_a[:,-1,1])**2 + np.mean(m_a[:,-1,2])**2)))
    
    #Plotting
    plotter_x = np.linspace(min(angles), max(angles), len(h_strengths))
    flopH_plotter = flopH*np.ones_like(h_strengths)
    plt.figure()
    plt.title("Angle between ground state and magnetic field")
    plt.xlabel("Magnetic field strength")
    plt.ylabel(r"$\theta$")
    plt.plot(h_strengths, angles, label =r"$\theta$")
    plt.plot(flopH_plotter,plotter_x, "--", color = "r", label ="Flop transistion")
    plt.legend()
    plt.show()

def hscanNew():
    """Similar to the previous function but with a different magnetic field 
    """
    #simulation constants
    L = 28
    dx = 1
    N = int(L/dx)
    N_steps = 1000
    current_length = int(0.1*N)
    od_length = int(0.2*N)
    L_y = 0
    Ny = int(L_y/dx)
    Ny = 0
    border_size = 0
    dt = 2.5*10**-3/config.w_m

    #Prepare physical constants
    alpha  = setAlpha(N,current_length, config.alph, config.alph, 5*config.alph, od_length=od_length, Ny = Ny) #Damping
    J = setInputCurrent(N,N_steps, current_length,od_length,from_step = 0, to_step = N_steps, Ny = Ny, on = True) #Input current
    d = np.array([0,0,0]) #DMI vector
    h = setHField(N,N_steps, Ny = Ny, dir = 0, hargs = (0.0)) #Make B
    h = setCircularField(N,N_steps, dir = 0, hargs = (100,10,0,11,150,dt))
    w_1 = np.zeros((3,3))
    w_2 = np.zeros((3,3), dtype = config.dtype)
    #Exchange
    D = 2
    w_ex = dx**2*6.4e2
    A = (1/(2*D))*w_ex
    A_1 = 0#w_ex*3/8
    A_2 = -w_ex/16#A_1 -A
    
    #Anisotropy
    #positive is hard axis!!!
    #5.4e-1
    w_1[2,2] = 5.4e-1
    w_1[0,0] = -2.4e-2
    
    #DMI
    D_bulk = 0
    d = np.array([0,0,0])

    beta = 2*10**-5*config.w_m*1000
    C = 0
    #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D)
    constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, J, d, h, D_bulk)

    #Set inital state
    init_state = initInGivenDirection(N, N_steps, Ny = Ny, initargs=([1,0.01,0]))
    
    flopH = np.sqrt(2.4*10**-2*(2*w_ex + 2.4*10**-2))
    max_h = flopH*2 #2*floph
    min_h = 0
    h_strengths = np.arange(min_h,max_h, 0.3)
    angles = np.zeros_like(h_strengths)

    for i in range(len(h_strengths)):
        h = setHField(N,N_steps, hargs = (h_strengths[i]))
        #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d, D)
        constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, J, d, h, D)
        m_a, m_b, _, _, _ = RK4.runSimulation(N,N_steps, dt, init_state, constants = constants)    
        angles[i] = np.arccos(np.sqrt(np.mean(m_a[:,-1,1])**2)/(np.sqrt(np.mean(m_a[:,-1,0])**2 + np.mean(m_a[:,-1,1])**2 + np.mean(m_a[:,-1,2])**2)))
    plotter_x = np.linspace(min(angles), max(angles), len(h_strengths))
    flopH_plotter = flopH*np.ones_like(h_strengths)
    plt.figure()
    plt.title("Angle between ground state and magnetic field")
    plt.xlabel("Magnetic field strength")
    plt.ylabel(r"$\theta$")
    plt.plot(h_strengths, angles, label =r"$\theta$")
    plt.plot(flopH_plotter,plotter_x, "--", color = "r", label ="Flop transistion")
    plt.legend()
    plt.show()

def hscanMagneticPhaseTransition():
    """
    A similar h-scan to the previous functions however this is done
    for a wider range of magnetic field strengths to observe 
    different magnetic phase transitions in a 1D system
    """
    #simulation constants
    L = 28
    dx = 1
    N = int(L/dx)
    N_steps = 50000
    current_length = int(0.1*N)
    od_length = int(0.2*N)
    L_y = 0
    Ny = int(L_y/dx)
    Ny = 0
    border_size = 0
    dt = 2.5*10**-3/config.w_m

    #Prepare physical constants
    alpha  = setAlpha(N,current_length, 10*config.alph, 10*config.alph, 10*config.alph, od_length=od_length, Ny = Ny) #Damping
    J = setInputCurrent(N,N_steps, current_length,od_length,from_step = 0, to_step = N_steps, Ny = Ny, on = False) #Input current
    h = setHField(N,N_steps, Ny = Ny, dir = 0, hargs = (0.0)) #Make B
    #h = setCircularField(N,N_steps, dir = 0, hargs = (100,10,0,11,150,dt))
    w_1 = np.zeros((3,3))
    w_2 = np.zeros((3,3), dtype = config.dtype)
    #Exchange
    D = 1
    w_ex = dx**2*6.4e2
    w_ex = w_ex*0.05
    A = (1/(2*D))*w_ex
    #A_1 = A
    #A_2 = 0
    A_2 = -0.25*A
    A_1 = A + A_2

    #Anisotropy
    #positive is hard axis!!!
    #5.4e-1
    #w_1[2,2] = 5.4e-1
    w_1[1,1] = -2.4e-2
    
    w_2 = 0.5*w_1


    #DMI
    D_bulk = 0

    d = np.array([0,0,0])

    beta = 2*10**-5*config.w_m*1000
    C = 0

    #Temp
    if Ny ==0:
        T = np.ones((N,3))
    else:
        T = np.ones((N, Ny,3))


    #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D, T)
    constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, J, d, h, D_bulk, T)

    #Set inital state
    init_state = initInGivenDirection(N, N_steps,initargs = ([0,1,0]))
    
    flopH = np.sqrt(2.4*10**-2*(2*w_ex + 2.4*10**-2))
    print(f'flopH = {flopH}')

    h_strengths = np.linspace(0,90,50)

    print('The different magnetic field: ', h_strengths)

    #Needed for plots
    angles = np.zeros_like(h_strengths)
    xvals = np.zeros_like(h_strengths)
    yvals = np.zeros_like(h_strengths)
    zvals = np.zeros_like(h_strengths)

    for i in range(len(h_strengths)):
        if h_strengths[i] < 22: #need more steps to relax for low magnetic fields
            N_steps = 70000
        else:
            N_steps = 50000
        J = setInputCurrent(N,N_steps, current_length,od_length,from_step = 0, to_step = N_steps, Ny = Ny, on = False) #Input current
        h = setHField(N,N_steps, hargs = (h_strengths[i]))
        init_state = initInGivenDirection(N, N_steps,initargs = ([0,1,0]))
        #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d, D)
        constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, J, d, h, D)
        m_a, m_b, _, _, _ = RK4.runSimulation(N,N_steps, dt, init_state, constants = constants)   
        """while not CheckRelaxed(m_a,m_b,tol = 1e-1):
            m_a[:,0,:] = m_a[:,-1,:]
            m_b[:,0,:] = m_b[:,-1,:]
            m_a, m_b, _, _, _ = RK4.runSimulation(N,N_steps, dt, (m_a, m_b), constants = constants) """
        xvals[i] = 0.5*(np.mean(m_a[:,-1,0]) + np.mean(m_b[:,-1,0]))
        yvals[i] = 0.5*(np.mean(m_a[:,-1,1]) + np.mean(m_b[:,-1,1]))
        zvals[i] = 0.5*(np.mean(m_a[:,-1,2]) + np.mean(m_b[:,-1,2]))
        angles[i] = np.arccos(np.sqrt(np.mean(m_a[:,-1,1])**2)/(np.sqrt(np.mean(m_a[:,-1,0])**2 + np.mean(m_a[:,-1,1])**2 + np.mean(m_a[:,-1,2])**2)))
        print(f'{i}\{len(h_strengths)}')

    fig, axs = plt.subplots(3,sharex=True)
    fig.suptitle('Magnetization as a function of applied Mag. field')
    axs[0].plot(h_strengths,xvals)
    axs[1].plot(h_strengths,yvals)
    axs[2].plot(h_strengths,zvals)
    plt.show()

    #The plot showing magnetizaiton as a function of magnetic field
    plt.figure()
    plt.plot(h_strengths,yvals)
    plt.xlabel('Mag.field strength B')
    plt.ylabel(r'$\langle M_A^y\rangle + \langle M_B^y \rangle$')
    plt.title('Magnetic Phase Transition')
    plt.show()
    np.save('h_strengthsA.npy', h_strengths)
    np.save('yvalsA.npy', yvals)

def alirezaTriplePlot():
    """The procedure for plotting the magnetization as a result of a
    particular input current. The simulation is run for a 1D system

    Returns:
        [type]: [description]
    """
    L = 100
    dx = 1
    N = int(L/dx)
    N_steps = 50000
    current_length = int(0.2*N)
    od_length = int(0.1*N)

    #Set anisotropy
    w_1 = np.zeros((3,3))
    #positive is hard axis!!!
    w_1[2,2] = 5.4e-1
    w_1[1,1] = -2.4e-2
    setAnisotropyMatrix(w_1)

    #Set alpha
    setAlpha(N,current_length, config.alph, 0.02, config.alph, od_length=od_length)

    print("w_m", config.w_m)
    beta = 2*10**-5*config.w_m*10
    print("beta", beta)
    dt = 5*10**-3/config.w_m
    #dt = 1*10**-4/w_m
    print("dt", dt)
    w_h = 6.6*config.gam/(2*np.pi*config.f_0)
    print(w_h)
    #Set J
    setInputCurrent(N,N_steps, current_length,from_step = 0, to_step = N_steps, on = True)

    m_a, m_b = RK4.main(N, N_steps, dt, grapher = True ,initialiser=randomizeStart, Hfunction= flopH, Zeeman_on=False, initargs=([0,1,0]))
    return 0

def test2D():
    """Procedure for running a 2D system
    """
    #simulation constants
    L_x = 20
    dx = 1
    N = int(L_x/dx)
    N_steps = 20000
    stride = 1
    current_length = int(0.1*N)
    od_length = int(0.2*N)
    L_y = 0
    Ny = int(L_y/dx)
    border_size = 0
    dt = 2.5*10**-3/config.w_m

    #Prepare physical constants
    alpha  = setAlpha(N,current_length, 5*config.alph, 10*config.alph, 10*config.alph, od_length=od_length, Ny = Ny) #Damping
    beta = 0.0
    C = 0
    J = setInputCurrent(N,N_steps, current_length,od_length,from_step = 0, to_step = N_steps, Ny = Ny, on = False) #Input current
    #d = np.array([0,0,0]) #DMI vector
    #h = setHField(N,N_steps, Ny = Ny, dir = 0, hargs = (0.0)) #Make B
    #h = setCircularField(N,N_steps, dir = 0, hargs = (100,10,0,11,15000,dt))
    h = setHField(N,N_steps, Ny = Ny, dir = 1, hargs = (0.0)) #Make B

    #(B0,omega,phi,Nt1,Nt2,dt)
    #h = setCircularField(N,N_steps, dir = 2, hargs = (4.5,10,0,11,0,dt))
    w_1 = np.zeros((3,3))
    #-0.5 of original one
    #Exchange
    D = 2
    w_ex = dx**2*6.4e2
    A = (1/(2*D))*w_ex
    A_1 = 0#w_ex*3/8
    A_2 = -w_ex/32#A_1 -A
    
    #Anisotropy
    #positive is hard axis!!!
    #5.4e-1
    w = np.zeros_like(w_1)
    w[2,2] = 5.4e-1
    w[0,0] = -2.4e-2
    w_1 = 0.5*w
    w_2 = w_1 - w
    #w_1[2,2] = 5.4e-1
    #w_1[0,0] = -2.4e-2
    
    #DMI
    D_bulk = 0
    d = np.array([1000,0,0])

    #Temp, do something with amp
    if Ny ==0:
        T = np.zeros((N,3))
    else:
        T = np.zeros((N, Ny,3))


    #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D, T)
    constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, J, d, h, D_bulk, T)
    

    #Set inital state
    init_state = initInGivenDirection(N,int(N_steps/stride),Ny=Ny,border_size=border_size, initargs=([0,1,0]))
    init_state = randomizeStart(N,N_steps,Ny = Ny)
    #Run simulations
    m_a, m_b, mdot_a, mdot_b, T = RK4.runSimulation(N, N_steps, dt, init_state,geom, border_size = border_size, Ny = Ny, constants = constants, cut_borders_at=N_steps, target = "cpu", stride = stride)
    
    
    
    #Call the plots you are interested in
    if Ny ==0:
        plot.Animate1D(N,int(N_steps/stride), m_a ,m_b)
        plot.Animate1D(N,N_steps//stride, (m_a +m_b)/2, (m_a+m_b)/2)
    else:
        plot.Animate2D(N,Ny, int(N_steps/stride), m_a, m_b)
        
    #plot.Animate1D(N,Ny, int(N_steps/stride), m_a, m_b)
    #plot.Animate2D(N,Ny, int(N_steps/stride), m_a-m_b, m_a-m_b)
    #plot.JOut(mdot_a, mdot_b, m_a, m_b, dt, T, beta, od_length, current_length)
    #hscan()
    #plot.normSurfacePlot(m_a,m_b)


def tempScan():
    w_h = 6.6*config.gam/(2*np.pi*config.f_0)


    #------------------SET FUNDAMENTAL SYSTEM CONSTANTS------------------
    g_e = 2.0
    alph = 0.00068
    mumag = 5.788e-5
    hbar = 6.582e-16
    D = 1
    d0 = 1
    dtype = np.float32
    gam = g_e * mumag / hbar
    
    w_ex2 = 1
    k_b = 1.380649e-23


    mu_0 = 4.0 * np.pi*10**-7
    M_s = 9.55e5
    f_0 = gam*mu_0*M_s
    d_0 = 4.17e-10
    t_0 = f_0**(-1)
    #r_0 = np.sqrt(w_ex*d_0**2/(2*D))
    J = 0
    d = np.array([0,0,0])
    C = 0
    w_m = 4*np.pi*mu_0*M_s*gam/f_0

    current_length = 0
    od_length = 0
    alpha_sp = 0.02
    delta = 4*3.549*2*d_0

    #SetSystemSizes
    Lx = 400e-9
    Ly = 180e-9
    dx = 1e-9



    #Gridsizes
    Nx,Ny = int(round(Ly/dx)),int(round(Lx/dx))    
    


    #Setting borders, overdamping regimes and current regimes
    border_size  = 0
    current_length = int(0.1*Nx)
    od_length = int(0.2*Nx)

    #RUNTIME CONSTANTS
    N_steps = 50000
    stride = 100

    
    border_size = 0
    dt = 2.5e-4/w_m


    alph = 0.0064
    #------------------PREPARE DAMPING TERM------------------
    alpha  = setAlpha(Nx,current_length, 10*alph, 10*alph, 10*alph, od_length=od_length, Ny = Ny) #Damping

    #------------------SET INPUT CURRENT------------------
    strength = 0.0 #Strength of current term
    beta = 1.0
    dir_J = 1.0
    J_args = (Nx, Ny, strength, dir_J) #Input current


    #------------------SET EXCHANGE------------------ 
    #100000 #Not sure about this, lattice points per step??(might make sense if we are talking units of d_0 or something, does not affect gradient and curl term, except ny adjusting strengths)
    #w_ex = #2e11*gam*dx**2/M_s
    w_ex = 6.4e2#*dx**2
    A = (1/(2*D))*w_ex#/dx**2 #Dx included here as well so it becomes dx**2/dx**2 which is accounted for later in the laplacian
    A_1 = w_ex*6/8
    A_2 = (A_1-A)

    #------------------SET MAGNETIC FIELD------------------
    omega_Z = 0
    dir_h = 0
    h_args = (Nx,Ny,omega_Z,dir_h)


    
    #------------------SET ANISOTROPY------------------
    #positive is hard axis!!!
    w = np.zeros((Nx,Ny,3,3), dtype = dtype)
    w_1 = np.zeros_like(w)
    w[:,:,0,0] = -2*2.54e-2#-0.54e6*gam/M_s
    w[:,:,2,2] = 2*5.4e-1#*10e6*gam/M_s
    w_1 = 0.8*w
    w_2 = w_1 - w #Define w2 as w1 - w

    #------------------DMI------------------
    D_bulk = 0#-10
    d = np.array([0,0,0], dtype = dtype)

    #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J_func,J_args,d,h_func,h_args,D,Temp)
    constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, Jfunc.constantJ,J_args, d, hfunc.constanth, h_args, D_bulk, 0)


    #Initialize geometry
    """geom = np.ones((Nx,Ny, 3), dtype = np.int64)
    geom[9,8:12,:] = 1
    geom = draw.drawTriangle(geom, 8,Ny,0)
    geom = draw.drawTriangle(geom, Nx-11, Ny,11)"""
    geom = imToArr.imToArrshape("Structures/Master_structure_1.png", Nx,Ny, show = False)

    init_state = initInGivenDirection(Nx,N_steps//stride,Ny=Ny,border_size=border_size, initargs=([1,0,0]))
    #convolve init state with that geometry
    init_state[0,:,:,0,:] = init_state[0,:,:,0,:]*geom
    init_state[1,:,:,0,:] = init_state[1,:,:,0,:]*geom
    
    #init_state = randomizeStart(N,N_steps//stride,Ny = Ny)


    #Get averaging constants
    min_temp = 3000
    max_temp = 4000
    Temps = np.arange(min_temp,max_temp,10)
    neel_avg_abs = np.zeros_like(Temps, dtype = np.float64)
    avg_length = 1
    neel_avg_helper = np.zeros(avg_length)
    def avger(avg_length, Temp):
        #Run simulations
        T = initConstTemp(N,Ny,Temp)
        J_args = (0,0,0,0)
        h_args = (0,0,0,0)
        init_state = initInGivenDirection(Nx,N_steps//stride,Ny=Ny,border_size=border_size, initargs=([1,0,0]))
        #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D, T)
        constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, Jfunc.constantJ, J_args, d, hfunc.constanth, h_args, D_bulk, T)

        #Run simulation
        m_a, m_b, mdot_a, mdot_b, T = RK4.runSimulation(Nx, N_steps, dt, init_state, geom,border_size = border_size, Ny = Ny, constants = constants, cut_borders_at=N_steps, target = "cpu", stride = stride, functional = True)    
        avg_absolute = np.sqrt(np.average(((m_a[:,:,-1,0] - m_b[:,:,-1,0]))/2, axis = (0,1))**2 + np.average(((m_a[:,:,-1,1] - m_b[:,:,-1,1]))/2, axis = (0,1))**2 + np.average(((m_a[:,:,-1,2] - m_b[:,:,-1,2]))/2, axis = (0,1))**2)
        
        print(avg_absolute)
        #neel_avg_helper = avg_absolute
        return avg_absolute
    
    for i in range(len(Temps)):
        for j in range((avg_length)):
            init_state = initInGivenDirection(Nx,N_steps//stride,Ny=Ny,border_size=border_size, initargs=([1,0,0]))
            print("Current Temperature Amplitude %d, temp %d of %d" % (Temps[i], min_temp, max_temp))
            #Run simulations
            T = initConstTemp(Nx,Ny,Temps[i])
            
            #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D, T)
            constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, Jfunc.constantJ,J_args, d, hfunc.constanth, h_args, D_bulk, T)

            #Run simulation
            m_a, m_b, mdot_a, mdot_b, T = RK4.runSimulation(Nx, N_steps, dt, init_state,geom, Ny = Ny, constants = constants, target = "cuda", stride = stride,functional = True)  
            m_a[m_a == 0] = np.nan            
            avg = np.sqrt(np.nanmean((m_a[:,:,-1,0] - m_b[:,:,-1,0])/2, axis = (0,1))**2 + np.nanmean((m_a[:,:,-1,1] - m_b[:,:,-1,1])/2, axis = (0,1))**2 + np.nanmean((m_a[:,:,-1,2] - m_b[:,:,-1,2])/2, axis = (0,1))**2)
            neel_avg_helper[j] = avg
        neel_avg_abs[i] = np.average(neel_avg_helper)
        print("Tot_avg: ", neel_avg_abs[i])
        np.savetxt("neel_avg_new.txt",neel_avg_abs)
        np.savetxt("Temps.txt",Temps)
        
    
    """for i in range(len(Temps)):
        results = Parallel(n_jobs = 2)(delayed(avger)(avg_length, Temps[i]) for j in range(avg_length))
        print("Current Temperature Amplitude %d of 1000" % (Temps[i]))
        print("Results from temperature %d \n" % (Temps[i]),results)
        neel_avg_abs[i] = np.average(results)"""
    np.savetxt("neel_avg_new.txt", neel_avg_abs)
    np.savetxt("Temps.txt", Temps)



def maiScan(fname, strength):
    #w_h = np.sqrt(2*abs(w_ex)*2.4e-2)
    #w_hc = np.sqrt(2.4*10**-2*(2*w_ex + 2.4*10**-2))
    w_h = 6.6*config.gam/(2*np.pi*config.f_0)


    #------------------SET FUNDAMENTAL SYSTEM CONSTANTS------------------
    g_e = 2.0
    alph = 0.00068
    mumag = 5.788e-5
    hbar = 6.582e-16
    D = 2
    d0 = 1
    dtype = np.float32
    gam = g_e * mumag / hbar
    
    w_ex2 = 1
    k_b = 1.380649e-23


    mu_0 = 4.0 * np.pi*10**-7
    M_s = 9.55e5
    f_0 = gam*mu_0*M_s
    d_0 = 4.17e-10
    t_0 = f_0**(-1)
    #r_0 = np.sqrt(w_ex*d_0**2/(2*D))
    J = 0
    d = np.array([0,0,0])
    C = 0
    w_m = 4*np.pi*mu_0*M_s*gam/f_0

    current_length = 0
    od_length = 0
    alpha_sp = 0.02
    delta = 4*3.549*2*d_0

    #SetSystemSizes
    Lx = 500e-9
    Ly = 180e-9
    dx = 2e-9



    #Gridsizes
    Nx,Ny = int(round(Ly/dx)),int(round(Lx/dx))    
    


    #Setting borders, overdamping regimes and current regimes
    border_size  = 0
    current_length = int(0.1*Nx)
    od_length = int(0.2*Nx)

    #RUNTIME CONSTANTS
    N_steps = 600000
    stride = 200
    
    border_size = 0
    dt = 5e-3/w_m


    alph = 1e-4
    od_alph = alph
    #------------------PREPARE DAMPING TERM------------------
    alpha  = ut.setAlpha(Nx,current_length, alph, alph, od_alph, od_length=50, Ny = Ny) #Damping

    #------------------SET INPUT CURRENT------------------
    from_ind = 50
    to_ind = 100
    strength = 0 #Strength of current term
    beta = 1.0
    dir_J = 0.0
    bg_strength = 0#0.165*strength
    timestep_stop = 200000
    J_args = (from_ind, to_ind, dir_J, strength, bg_strength, timestep_stop) #Input current


    #------------------SET EXCHANGE------------------ 
    #100000 #Not sure about this, lattice points per step??(might make sense if we are talking units of d_0 or something, does not affect gradient and curl term, except ny adjusting strengths)
    #w_ex = #2e11*gam*dx**2/M_s
    w_ex = 0.1*6.4e2#*dx**2
    A = (d_0**2/(2*D*dx**2))*w_ex#/dx**2 #Dx included here as well so it becomes dx**2/dx**2 which is accounted for later in the laplacian
    A_1 = w_ex*6/8
    A_2 = (A_1-A)

    #------------------SET MAGNETIC FIELD------------------
    strength_h = strength
    dir_h = 0
    start_index = 110
    end_index = 140
    """start_i = h[0]
    end_i = h[1]
    dir_h = h[3]
    strength = h[2]"""
    h_args = (start_index,end_index,strength_h,dir_h)


    
    #------------------SET ANISOTROPY------------------
    #positive is hard axis!!!
    w = np.zeros((Nx,Ny,3,3), dtype = dtype)
    w_1 = np.zeros_like(w)
    easy_amplitude = -0.01#Put this to 0 at 25.04, seems to almost work, also bulk_easy makes a single spike switch of 
    #material
    hard_amp = 1
    w[:,40:110,1,1] = easy_amplitude*5.4e-1#-2*2.54e-2#-0.54e6*gam/M_s
    w[:,40:110,0,0] = hard_amp*5.4e-1#*10e6*gam/M_s
    w[:,140:210,1,1] = easy_amplitude*5.4e-1#-2*2.54e-2#-0.54e6*gam/M_s
    w[:,140:210,0,0] = hard_amp*5.4e-1#*10e6*gam/M_
    
    bulk_coef_hard = 1
    bulk_easy = -0.01#-0.0245#-0.023 is very close
    w[:,:40,0,0] = bulk_coef_hard*5.4e-1
    w[:,110:140,0,0] = bulk_coef_hard*5.4e-1
    w[:,210:,0,0] = bulk_coef_hard*5.4e-1
    w[:,:40,1,1] = bulk_easy*5.4e-1
    w[:,110:140,1,1] = bulk_easy*5.4e-1
    w[:,210:,1,1] = bulk_easy*5.4e-1
    #w[:,not 50:150 and not 250:350,1,1] = 5.4e-1
    #w[:,:,1,1] = 5.4e-1    
    w_1 = 0.8*w
    w_2 = w_1 - w #Define w2 as w1 - w

    #------------------DMI------------------
    D_bulk = 0#-10
    d = np.array([-0.005*5.4e-1,0,0], dtype = dtype)

    #Temperature
    
    Temp = 0
    Temp_constant = 0#(alpha/M_s)*((k_b*Temp)/gam)
    Temp = ut.initConstTemp(Nx,Ny,Temp_constant)

    #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J_func,J_args,d,h_func,h_args,D,Temp)
    constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, Jfunc.constantJ,J_args, d, hfunc.constanth, h_args, D_bulk, Temp)

    #SET INITIAL STATE OF THE SYSTEM
    #init_state = ut.initInGivenDirection(Nx,int(N_steps/stride),Ny=Ny,border_size=border_size, initargs=([0,1,0]))
    #init_state = ut.initExpSetup(Nx,Ny,int(N_steps//stride))
    #init_state = np.zeros((2,Nx,Ny,int(N_steps/stride),3))
    #init_state[:,:,:,0,:] = np.array([np.load(r"GS_setup_new_a.npy"),np.load(r"GS_setup_new_b.npy")])
    init_state = ut.randomizeStart(Nx,N_steps//stride,Ny = Ny)


    #Initialize geometry
    """geom = np.ones((Nx,Ny, 3), dtype = np.int64)
    geom[9,8:12,:] = 1
    geom = draw.drawTriangle(geom, 8,Ny,0)
    geom = draw.drawTriangle(geom, Nx-11, Ny,11)"""
    geom = imToArr.imToArrshape("Structures/Master_structure_1_tail.png", Nx,Ny, show = False)


    #convolve init state with that geometry
    init_state[0,:,:,0,:] = init_state[0,:,:,0,:]*geom
    init_state[1,:,:,0,:] = init_state[1,:,:,0,:]*geom
    
    """plt.imshow(init_state[0,:,:,0,0])
    plt.show()
    """
    print("dt: ", dt)
    print("w_m: ", w_m)#Printing sys size
    print("System constants:\n Nx: %i    Ny: %i    N_steps:   %i    stride:     %i" %(Nx,Ny,N_steps,stride))
    print("Exchange coupling parameters: \n w_ex:" ,w_ex," A:    " , A)
    #unit conversion consants
    A_tilde = d_0**2*w_ex/(2*D)
    r_0 = np.sqrt(A_tilde*t_0)
    t_0 = (gam*mu_0*M_s)**-1
    #print(t_0)
    print("Length:" , Lx, "by ",Ly, "Total time:", dt*N_steps*t_0)
    
    m_a, m_b, mdot_a, mdot_b, T = RK4.runSimulation(Nx, N_steps, dt, init_state,geom, Ny = Ny, constants = constants, target = "cuda", stride = stride, functional = True)
    T = (T*t_0)
    
    np.save("GS_setup_new_a.npy", m_a[:,:,-1,:])
    np.save("GS_setup_new_b.npy", m_b[:,:,-1,:])

    #Plotting
    plot.SpinAnimationWrapper(m_a, m_b, Nx, N_steps, stride, Ny, steps_along_axis = 5, average = False) 
    
    in_neel = (m_a[45,75,:,:] - m_b[45,75,:,:])/2
    out_neel = (m_a[45,175,:,:]- m_b[45,175,:,:])/2
    in_phi = np.arctan2(in_neel[:,2],in_neel[:,1])
    out_phi = np.arctan2(out_neel[:,2],out_neel[:,1])

    #w_h = np.sqrt(2*abs(w_ex)*2.4e-2)
    #w_hc = np.sqrt(2.4*10**-2*(2*w_ex + 2.4*10**-2))
    w_h = 6.6*config.gam/(2*np.pi*config.f_0)


    #------------------SET FUNDAMENTAL SYSTEM CONSTANTS------------------
    g_e = 2.0
    alph = 0.00068
    mumag = 5.788e-5
    hbar = 6.582e-16
    D = 2
    d0 = 1
    dtype = np.float32
    gam = g_e * mumag / hbar
    
    w_ex2 = 1
    k_b = 1.380649e-23


    mu_0 = 4.0 * np.pi*10**-7
    M_s = 9.55e5
    f_0 = gam*mu_0*M_s
    d_0 = 4.17e-10
    t_0 = f_0**(-1)
    #r_0 = np.sqrt(w_ex*d_0**2/(2*D))
    J = 0
    d = np.array([0,0,0])
    C = 0
    w_m = 4*np.pi*mu_0*M_s*gam/f_0

    current_length = 0
    od_length = 0
    alpha_sp = 0.02
    delta = 4*3.549*2*d_0

    #SetSystemSizes
    Lx = 500e-9
    Ly = 180e-9
    dx = 2e-9



    #Gridsizes
    Nx,Ny = int(round(Ly/dx)),int(round(Lx/dx))    
    


    #Setting borders, overdamping regimes and current regimes
    border_size  = 0
    current_length = int(0.1*Nx)
    od_length = int(0.2*Nx)

    #RUNTIME CONSTANTS
    N_steps = 600000
    stride = 200
    
    border_size = 0
    dt = 5e-3/w_m


    alph = 1e-4
    od_alph = alph
    #------------------PREPARE DAMPING TERM------------------
    alpha  = ut.setAlpha(Nx,current_length, alph, alph, od_alph, od_length=50, Ny = Ny) #Damping

    #------------------SET INPUT CURRENT------------------
    from_ind = 50
    to_ind = 100
    strength = 32 #Strength of current term
    beta = 1.0
    dir_J = 0.0
    bg_strength = 0#0.165*strength
    timestep_stop = 200000
    J_args = (from_ind, to_ind, dir_J, strength, bg_strength, timestep_stop) #Input current


    #------------------SET EXCHANGE------------------ 
    #100000 #Not sure about this, lattice points per step??(might make sense if we are talking units of d_0 or something, does not affect gradient and curl term, except ny adjusting strengths)
    #w_ex = #2e11*gam*dx**2/M_s
    w_ex = 0.1*6.4e2#*dx**2
    A = (d_0**2/(2*D*dx**2))*w_ex#/dx**2 #Dx included here as well so it becomes dx**2/dx**2 which is accounted for later in the laplacian
    A_1 = w_ex*6/8
    A_2 = (A_1-A)

    #------------------SET MAGNETIC FIELD------------------
    strength_h = strength
    dir_h = 0
    start_index = 110
    end_index = 140
    """start_i = h[0]
    end_i = h[1]
    dir_h = h[3]
    strength = h[2]"""
    h_args = (start_index,end_index,strength_h,dir_h)
    """start_i = h[0]
    end_i = h[1]
    dir_h = h[3]
    strength = h[2]"""


    
    #------------------SET ANISOTROPY------------------
    #positive is hard axis!!!
    w = np.zeros((Nx,Ny,3,3), dtype = dtype)
    w_1 = np.zeros_like(w)
    easy_amplitude = -0.01#Put this to 0 at 25.04, seems to almost work, also bulk_easy makes a single spike switch of 
    #material
    hard_amp = 1
    w[:,40:110,1,1] = easy_amplitude*5.4e-1#-2*2.54e-2#-0.54e6*gam/M_s
    w[:,40:110,0,0] = hard_amp*5.4e-1#*10e6*gam/M_s
    w[:,140:210,1,1] = easy_amplitude*5.4e-1#-2*2.54e-2#-0.54e6*gam/M_s
    w[:,140:210,0,0] = hard_amp*5.4e-1#*10e6*gam/M_
    
    bulk_coef_hard = 1
    bulk_easy = -0.01#-0.0245#-0.023 is very close
    w[:,:40,0,0] = bulk_coef_hard*5.4e-1
    w[:,110:140,0,0] = bulk_coef_hard*5.4e-1
    w[:,210:,0,0] = bulk_coef_hard*5.4e-1
    w[:,:40,1,1] = bulk_easy*5.4e-1
    w[:,110:140,1,1] = bulk_easy*5.4e-1
    w[:,210:,1,1] = bulk_easy*5.4e-1
    #w[:,not 50:150 and not 250:350,1,1] = 5.4e-1
    #w[:,:,1,1] = 5.4e-1    
    w_1 = 0.8*w
    w_2 = w_1 - w #Define w2 as w1 - w

    #------------------DMI------------------
    D_bulk = 0#-10
    d = np.array([-0.005*5.4e-1,0,0], dtype = dtype)

    #Temperature
    
    Temp = 0
    Temp_constant = 0#(alpha/M_s)*((k_b*Temp)/gam)
    Temp = ut.initConstTemp(Nx,Ny,Temp_constant)

    #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J_func,J_args,d,h_func,h_args,D,Temp)
    constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, Jfunc.constantJ,J_args, d, hfunc.constanth, h_args, D_bulk, Temp)

    #SET INITIAL STATE OF THE SYSTEM
    #init_state = ut.initInGivenDirection(Nx,int(N_steps/stride),Ny=Ny,border_size=border_size, initargs=([0,1,0]))
    #init_state = ut.initExpSetup(Nx,Ny,int(N_steps//stride))
    init_state = np.zeros((2,Nx,Ny,int(N_steps/stride),3))
    init_state[:,:,:,0,:] = np.array([np.load(r"GS_setup_new_a.npy"),np.load(r"GS_setup_new_b.npy")])
    #init_state = ut.randomizeStart(Nx,N_steps//stride,Ny = Ny)


    #Initialize geometry
    """geom = np.ones((Nx,Ny, 3), dtype = np.int64)
    geom[9,8:12,:] = 1
    geom = draw.drawTriangle(geom, 8,Ny,0)
    geom = draw.drawTriangle(geom, Nx-11, Ny,11)"""
    geom = imToArr.imToArrshape("Structures/Master_structure_1_tail.png", Nx,Ny, show = False)


    #convolve init state with that geometry
    init_state[0,:,:,0,:] = init_state[0,:,:,0,:]*geom
    init_state[1,:,:,0,:] = init_state[1,:,:,0,:]*geom
    
    """plt.imshow(init_state[0,:,:,0,0])
    plt.show()
    """
    print("dt: ", dt)
    print("w_m: ", w_m)#Printing sys size
    print("System constants:\n Nx: %i    Ny: %i    N_steps:   %i    stride:     %i" %(Nx,Ny,N_steps,stride))
    print("Exchange coupling parameters: \n w_ex:" ,w_ex," A:    " , A)
    #unit conversion consants
    A_tilde = d_0**2*w_ex/(2*D)
    r_0 = np.sqrt(A_tilde*t_0)
    t_0 = (gam*mu_0*M_s)**-1
    #print(t_0)
    print("Length:" , Lx, "by ",Ly, "Total time:", dt*N_steps*t_0)
    
    m_a, m_b, mdot_a, mdot_b, T = RK4.runSimulation(Nx, N_steps, dt, init_state,geom, Ny = Ny, constants = constants, target = "cuda", stride = stride, functional = True)
    T = (T*t_0)
    
    #np.save("GS_setup_new_a.npy", m_a[:,:,:,:])
    #np.save("GS_setup_new_b.npy", m_b[:,:,:,:])

    #Plotting
    plot.SpinAnimationWrapper(m_a, m_b, Nx, N_steps, stride, Ny, steps_along_axis = 5, average = False) 
    
    in_neel = (m_a[45,75,:,:] - m_b[45,75,:,:])/2
    out_neel = (m_a[45,175,:,:]- m_b[45,175,:,:])/2
    in_phi = np.arctan2(in_neel[:,2],in_neel[:,1])
    out_phi = np.arctan2(out_neel[:,2],out_neel[:,1])
    np.save(fname + "Out_phi.npy",out_phi)
    np.save(fname + "in_phi.npy",in_phi)

def currentAnalyzer(currents,d_on):
    #------------------SET FUNDAMENTAL SYSTEM CONSTANTS------------------
    g_e = 2.0
    alph = 0.00068
    mumag = 5.788e-5
    hbar = 6.582e-16
    D = 2
    d0 = 1
    dtype = np.float32
    gam = g_e * mumag / hbar
    
    w_ex2 = 1
    k_b = 1.380649e-23


    mu_0 = 4.0 * np.pi*10**-7
    M_s = 9.55e5
    f_0 = gam*mu_0*M_s
    d_0 = 4.17e-10
    t_0 = f_0**(-1)
    #r_0 = np.sqrt(w_ex*d_0**2/(2*D))
    J = 0
    d = np.array([0,0,0])
    C = 0
    w_m = 4*np.pi*mu_0*M_s*gam/f_0
    constants = tuple(np.load("standard_biaxial_setup.npy", allow_pickle = True))
    
    Lx = 500e-9
    Ly = 180e-9
    dx = 2e-9
    #Gridsizes
    Nx,Ny = int(round(Ly/dx)),int(round(Lx/dx))    
    #Setting borders, overdamping regimes and current regimes
    border_size  = 0
    current_length = int(0.1*Nx)
    od_length = int(0.2*Nx)
    dt = 5e-3/w_m
    #RUNTIME CONSTANTS
    N_steps = 600000
    stride = 200
    init_state = ut.initInGivenDirection(Nx,int(N_steps/stride),Ny=Ny,border_size=border_size, initargs=([0,1,0]))


    #Initialize geometry
    """geom = np.ones((Nx,Ny, 3), dtype = np.int64)
    geom[9,8:12,:] = 1
    geom = draw.drawTriangle(geom, 8,Ny,0)
    geom = draw.drawTriangle(geom, Nx-11, Ny,11)"""
    geom = imToArr.imToArrshape("Structures/Master_structure_1_tail.png", Nx,Ny, show = False)
    #convolve init state with that geometry
    init_state[0,:,:,0,:] = init_state[0,:,:,0,:]*geom
    init_state[1,:,:,0,:] = init_state[1,:,:,0,:]*geom
    for i in range(len(currents)):
        #constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, Jfunc.constantJ,J_args, d, hfunc.constanth, h_args, D_bulk, Temp)
        w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, Jfunc.constantJ,J_args, d, hfunc.constanth, h_args, D_bulk, Temp = constants
        
        from_ind = 50
        to_ind = 100
        dir = 0.0
        strength = -1*currents[i]
        bias_strength = 0
        timestep_stop = N_steps
        
        J_args = (from_ind, to_ind, dir, strength, bias_strength, timestep_stop)
        if not d_on:
            d = np.zeros(3)
        constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, Jfunc.constantJ,J_args, d, hfunc.constanth, h_args, D_bulk, Temp)

        m_a, m_b, mdot_a, mdot_b, T = RK4.runSimulation(Nx, N_steps, dt, init_state,geom, Ny = Ny, constants = constants, target = "cuda", stride = stride, functional = True)
        #np.save("GS_setup_new_a.npy", m_a[:,:,:,:])
        #np.save("GS_setup_new_b.npy", m_b[:,:,:,:])
        in_neel_1 = (m_a[45,75,:,:] - m_b[45,175,:,:])/2
        out_neel_1 = (m_a[45,175,:,:] - m_b[45,175,:,:])/2
        #out_neel_2 = (m_a[33,175,:,:]- m_b[33,175,:,:])/2
        if np.isnan(in_neel_1).any() or np.isnan(out_neel_1).any():
            print("Støtte på nan ved denne styrken", currents[i])
        in_phi_1 = np.arctan2(in_neel_1[:,2],in_neel_1[:,1])
        #out_phi_2 = np.arctan2(out_neel_2[:,2],out_neel_2[:,1])
        out_phi_1 = np.arctan2(out_neel_1[:,2],out_neel_1[:,1])
        if not d_on:
            np.save("data/Current_data_without_dmi/Out_Neel" + str(currents[i]) + ".npy",out_neel_1)
            np.save("data/Current_data_without_dmi/in_Neel" + str(currents[i]) + ".npy",in_neel_1)
        else:
            np.save("data/Current_data_with_dmi/Out_Neel" + str(currents[i]) + ".npy",out_neel_1)
            np.save("data/Current_data_with_dmi/in_Neel" + str(currents[i]) + ".npy",in_neel_1)

if __name__ == "__main__":
    strengths = np.arange(1,10,1)
    for i in range(10):
        maiScan(str(i), strengths[i])
    