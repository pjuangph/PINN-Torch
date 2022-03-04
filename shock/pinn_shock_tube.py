import sys, torch
import math
from typing import Dict
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0,'../../nangs')
from burgers_lib import create_domain
from ml.burgers_pde import burgers_pde
from nangs import Dirichlet, MLP 
from nangs.samplers import RandomSampler
import numpy as np 
import json 
import matplotlib.pyplot as plt 
from matplotlib import cm

device = "cuda" if torch.cuda.is_available() else "cpu"

with open('settings.json','r') as f:
    settings = json.load(f)
    Ls = settings['general']['xmax']
    tmax = settings['general']['tmax']
    x = np.linspace(0,Ls,settings['general']['nx'])

    left_side = settings['initial_conditions']['left-side']
    right_side = settings['initial_conditions']['right-side']

    # Reference Conditions 
    P0_s = (left_side['P0']+right_side['P0'])/2.0  # Reference Pressure Pa
    T0_s = (left_side['T0']+right_side['T0'])/2.0  # Kelvin
    rho_s =  P0_s / ( R * T0_s )
    u_s = sqrt(gamma*R*T0_s) # Reference velocity is speed of sound (m/s)
    Cv = air.cv(T=T0_s,p=P0_s/101325)*1000 # 101325 - converts to bars. 1000 - converts to J
    Cp_ref = air.cp(T=T0_s,p=P0_s/101325)*1000
    e_s = Cv[0]*T0_s               # Internal Energy
    E_s = rho_s*(e_s+u_s*u_s/2) # Total Energy
    mu_s = sutherland(T0_s)
    Re_s = u_s * rho_s * Ls / mu_s

    # Initialize 
    # Note: Total = static at t=0, velocity = 0 
    ind = np.where(x < settings['general']['diaphram-xloc'])
    ## Left side 
    P0 = left_side['P0'] 
    T0 = left_side['T0']    # Kelvin
    rho =  P0 / ( R * T0 )
    Cv = air.cv(T=left_side['T0'],p=left_side['P0']/101325) * 1000
    Cp_left = air.cp(T=left_side['T0'],p=left_side['P0']/101325) * 1000
    u = 0
    E = rho*(Cv*T0+0.5*u*u)

    v[0,ind] = rho
    v[1,ind] = rho*u
    v[2,ind] = E # Velocity is 0 so Total energy = internal energy + kinetic
    Cp[ind] = Cp_left