import numpy as np

g_e = 2.0
alph = 0.0068
w_ex = 6.4*10**2
mumag = 5.788e-5
hbar = 6.582e-16
D = 1
d0 = 1
dtype = np.float32
gam = g_e * mumag / hbar
w_ex2 = 1
A = (1/(2*D))*w_ex
A_1 = 0#w_ex/10#A
A_2 = -w_ex/8#A_1 -A
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
d = np.array([1,0,0])
w_m = 4*np.pi*mu_0*M_s*gam/f_0
current_length = 0
od_length = 0
alpha_sp = 0.02
delta = 4*3.549*2*d_0