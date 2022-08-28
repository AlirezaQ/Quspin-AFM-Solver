import sys
sys.path.append("src")
import RK4 as RK4
import numpy as np 
import plotting as plot
import utilities as ut
import matplotlib
import matplotlib.pyplot as plt
import config
from mpl_toolkits import mplot3d
import matplotlib
import JFunctions as Jfunc
import hFunctions as hfunc
import drawGeom as draw
import imageToNdarray as imToArr
import scans

"""from accelerate import profiler"""

plt.switch_backend('Qt5Agg')
dtype = np.float32


def nonFunctionalTester():
    #w_h = np.sqrt(2*abs(w_ex)*2.4e-2)
    #w_hc = np.sqrt(2.4*10**-2*(2*w_ex + 2.4*10**-2))
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



    mu_0 = 4.0 * np.pi*10**-7
    M_s = 1.94e5
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


    #SYSTEM CONSTANTS
    L_x = 100
    dx = 1 #All calcuations use dx = 1 because the calcs are done with unitless constants
    Nx = int(L_x/dx)
    L_y = 100 #Set to zero if you want 1D
    Ny = int(L_y/dx)

    #Setting borders, overdamping regimes and current regimes
    border_size  = 0
    current_length = 0#int(0.1*Nx)
    od_length = 0#int(0.2*Nx)

    #RUNTIME CONSTANTS
    N_steps = 10000
    stride = 100

    border_size = 0
    dt = 2.5*10**-3/w_m


    #------------------PREPARE DAMPING TERM------------------
    alpha  = ut.setAlpha(Nx,current_length, 10*alph, 15*alph, 15*alph, od_length=od_length, Ny = Ny) #Damping

    #------------------SET INPUT CURRENT------------------
    beta = 0.0 #Strength of current term
    J = ut.setInputCurrent(Nx,N_steps, current_length,od_length,from_step = 0, to_step = N_steps, Ny = Ny, on = False) #Input current

    #------------------SET MAGNETIC FIELD------------------
    #h = ut.setHField(Nx,N_steps, Ny = Ny, dir = 0, hargs = (0.0)) #Make B
    #h = ut.setCircularField(Nx,N_steps, dir = 0, hargs = (100,10,0,11,15000,dt))
    h = ut.setHField(Nx,N_steps, Ny = Ny, dir = 1, hargs = (0.0)) #Make B

    #------------------SET EXCHANGE------------------
    D = 2
    w_ex = 6.4e2#dx**2*6.4e2
    A = (1/(2*D))*w_ex/dx**2 # dx^2 is indirectly accounted for in the laplacian later, so it becomes wrong to include it here.
    A_1 = w_ex*3/8
    A_2 = A_1 -A
    
    #------------------SET ANISOTROPY------------------
    #positive is hard axis!!!
    w = np.zeros((Nx,Ny,3,3))
    w_1 = np.zeros_like(w)
    w[:,:,0,0] = -2.4e-2
    w[:,:,2,2] = 5.4e-1
    w_1 = 0.5*w
    w_2 = w_1 - w #Define w2 as w1 - w

    #------------------DMI------------------
    D_bulk = 0#-10
    d = np.array([0,0,0], dtype = dtype)

    #Temperature
    Temperature_amp = 0
    Temp = ut.initConstTemp(Nx,Ny,Temperature_amp)


    #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J,d,h,D,Temp)
    constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, J, d, h, D_bulk, Temp)

    #SET INITIAL STATE OF THE SYSTEM
    init_state = ut.initInGivenDirection(Nx,int(N_steps/stride),Ny=Ny,border_size=border_size, initargs=([0,1,0]))
    #init_state = ut.randomizeStart(Nx,N_steps//stride,Ny = Ny)

    
    #Initialize geometry
    """geom = np.ones((Nx,Ny, 3), dtype = np.int64)
    geom[9,8:12,:] = 1
    geom = draw.drawTriangle(geom, 8,Ny,0)
    geom = draw.drawTriangle(geom, Nx-11, Ny,11)"""
    geom = imToArr.imToArrshape("Structures/Master_structure_1_2.png", Nx,Ny)

    #convolve init state with that geometry
    init_state[0,:,:,0,:] = init_state[0,:,:,0,:]*geom
    init_state[1,:,:,0,:] = init_state[1,:,:,0,:]*geom
    
    
    print("dt", dt)
    print("w_m", w_m)
    m_a, m_b, mdot_a, mdot_b, T = RK4.runSimulation(Nx, N_steps, dt, init_state,geom,border_size = border_size, Ny = Ny, constants = constants, cut_borders_at=N_steps, target = "cuda", stride = stride, functional = False)
    T = (T/f_0)*10**9
    #Plotting

    plot.SpinAnimationWrapper(m_a, m_b, Nx, N_steps, stride, Ny, steps_along_axis = 2, average = False)
    plt.show() 
    plot.normSurfacePlot(m_a, m_b)



def functionalMethodTester(curr = -32, d_on = True, bg_curr = 0):
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
    alpha  = ut.setAlpha(Nx,current_length, alph, alph, od_alph, Ny = Ny) #Damping

    #------------------SET INPUT CURRENT------------------
    from_ind = 50
    to_ind = 100
    strength = -32#Strength of current term
    beta = 1.0
    dir_J = 0.0
    bg_strength = bg_curr#s0.6*strength
    timestep_stop = 100000
    J_args = (from_ind, to_ind, dir_J, strength, bg_strength, timestep_stop) #Input current


    #------------------SET EXCHANGE------------------ 
    #100000 #Not sure about this, lattice points per step??(might make sense if we are talking units of d_0 or something, does not affect gradient and curl term, except ny adjusting strengths)
    #w_ex = #2e11*gam*dx**2/M_s
    w_ex = 0.1*6.4e2#*dx**2
    A = (d_0**2/(2*D*dx**2))*w_ex#/dx**2 #Dx included here as well so it becomes dx**2/dx**2 which is accounted for later in the laplacian
    A_1 = w_ex*6/8
    A_2 = (A_1-A)

    #------------------SET MAGNETIC FIELD------------------
    strength_h = -6*1.50#-10
    dir_h = 0
    start_index = 0
    end_index = 280
    """start_i = h[0]
    end_i = h[1]
    dir_h = h[3]
    strength = h[2]"""
    h_args = (start_index,end_index,strength_h,dir_h)


    
    #------------------SET ANISOTROPY------------------
    #positive is hard axis!!!
    w = np.zeros((Nx,Ny,3,3), dtype = dtype)
    w_1 = np.zeros_like(w)
    #Put this to 0 at 25.04, seems to almost work, also bulk_easy makes a single spike switch of 
    #material As of now it works
    hard_amp = 1
    easy_amplitude = -0.01
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
    d_amp = 0.005*5.4e-1
    d = np.array([d_amp,0,0], dtype = dtype)

    #Temperature
    
    Temp = 0
    Temp_constant = 0#(alpha/M_s)*((k_b*Temp)/gam)
    Temp = ut.initConstTemp(Nx,Ny,Temp_constant)

    #load constants to tuple of shape (w_ex,A_1,A_2,w_1,w_2,beta,C,alpha,J_func,J_args,d,h_func,h_args,D,Temp)
    
    #np.save("standard_biaxial_setup.npy", np.array(constants, dtype = object))
    #constants = tuple(np.load("standard_biaxial_setup.npy", allow_pickle = True))
    #SET INITIAL STATE OF THE SYSTEM
    init_state = ut.initInGivenDirection(Nx,int(N_steps/stride),Ny=Ny,border_size=border_size, initargs=([1,0,0]))
    #init_state = ut.initExpSetup(Nx,Ny,int(N_steps//stride))
    #init_state = np.zeros((2,Nx,Ny,int(N_steps/stride),3))
    #init_state[:,:,:,0,:] = np.array([np.load(r"GS_setup_new_a.npy"),np.load(r"GS_setup_new_b.npy")])
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
    

    from_ind = 50
    to_ind = 100
    strength = curr#-1*currents[i]#Strength of current term
    beta = 1.0
    dir_J = 0.0
    bg_strength = bg_curr#s0.6*strength
    timestep_stop = 100000
    J_args = (from_ind, to_ind, dir_J, strength, bg_strength, timestep_stop)
    if not d_on:
        d = np.zeros(3)
    constants = (w_ex, A_1, A_2, w_1, w_2, beta, C, alpha, Jfunc.constantJ,J_args, d, hfunc.constanth, h_args, D_bulk, Temp)
    init_state = ut.initInGivenDirection(Nx,int(N_steps/stride),Ny=Ny,border_size=border_size, initargs=([0,1,0]))
    #init_state = ut.initExpSetup(Nx,Ny,int(N_steps//stride))
    #init_state = np.zeros((2,Nx,Ny,int(N_steps/stride),3))
    #init_state[:,:,:,0,:] = np.array([np.load(r"GS_setup_new_a.npy"),np.load(r"GS_setup_new_b.npy")])
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
    print(curr, bg_curr)
    m_a, m_b, mdot_a, mdot_b, T = RK4.runSimulation(Nx, N_steps, dt, init_state,geom, Ny = Ny, constants = constants, target = "cuda", stride = stride, functional = True)
    T = (T*t_0)
        
    """np.save("GS_setup_new_a.npy", m_a[:,:,-1,:])
    np.save("GS_setup_new_b.npy", m_b[:,:,-1,:])"""

    #Plotting
    plot.SpinAnimationWrapper(m_a, m_b, steps_along_axis = 5)
        
    in_neel_1 = (m_a[45,75,:,:] - m_b[45,75,:,:])/2
    out_neel_1 = (m_a[45,175,:,:] - m_b[45,175,:,:])/2
    #out_neel_2 = (m_a[33,175,:,:]- m_b[33,175,:,:])/2
    in_phi_1 = np.arctan2(in_neel_1[:,2],in_neel_1[:,1])
    #out_phi_2 = np.arctan2(out_neel_2[:,2],out_neel_2[:,1])
    out_phi_1 = np.arctan2(out_neel_1[:,2],out_neel_1[:,1])
    #np.save("data\\bg_curr_data\\" + str(bg_strength) +"m_a.npy", m_a)
    #np.save("data\\bg_curr_data\\" + str(bg_strength) +"m_b.npy", m_b)
    #np.save("in_neel.npy", in_neel_1)
    #np.save("out_neel.npy", out_neel_1)
    plt.figure()
    plt.plot(np.arange(0,N_steps,stride),np.sin((in_phi_1))**2, label = r"$\sin{\frac{\phi_0}{2}}^2$")
    plt.plot(np.arange(0,N_steps,stride),np.sin((out_phi_1))**2, label = r"$\sin{\frac{\phi_1}{2}}^2$")
    #plt.plot(np.arange(0,N_steps,stride),np.sin((out_phi_2))**2, label = r"$\sin{\frac{\phi_2}{2}}^2$")
    plt.legend()
    plt.show()
    


    


if __name__ == '__main__':

    functionalMethodTester()

     
#TODO: Check out this: https://curiouscoding.nl/phd/2021/03/24/numba-cuda-speedup/ for possible speedups