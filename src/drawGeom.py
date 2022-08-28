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



def drawTriangle(state, base, height,start_x):
    for i in range(base):
        for j in range(height):
            state[i:height- i,start_x +i,:] = 1
    return state