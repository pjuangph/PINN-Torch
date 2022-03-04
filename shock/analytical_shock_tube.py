from cmath import nan
from math import sqrt
import pickle
import numpy as np 
import json 
import pyromat as pm
from fluids import sutherland
air = pm.get('ig.air')
with open('settings.json','r') as f:
    settings = json.load(f)

settings['initial_conditions']['left-side']['P0'] *= 101325 # Convert to Pa
settings['initial_conditions']['right-side']['P0'] *= 101325
R = settings['general']['IdealGasConstR']
gamma = settings['general']['gamma']
tmax = settings['general']['tmax']

'''
    Shock Tube Solution
'''
# Allocate memory
Ls = settings['general']['xmax']
tmax = settings['general']['tmax']
x = np.linspace(0,Ls,settings['general']['nx'])
Cp = x*0
dt = settings['general']['dt']
dx = x[1]-x[0]
v = np.zeros(shape=(3,len(x))) # rho 
                               # rho*v
                               # e
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
## Right side 
ind = np.where(x >= settings['general']['diaphram-xloc'])
right_side = settings['initial_conditions']['right-side']
P0 = right_side['P0']   # Pa
T0 = right_side['T0']   # K 
rho =  P0 / ( R * T0 )
u = 0
Cv = air.cv(T=right_side['T0'],p=right_side['P0']/101325)*1000 # Need pressure in bar 
Cp_right = air.cp(T=right_side['T0'],p=right_side['P0']/101325) * 1000
E = rho*(Cv*T0+0.5*u*u)

v[0,ind] = rho
v[1,ind] = rho*u
v[2,ind] = E
Cp[ind] = Cp_right
# Solve 
history = list() 
history.append({'t':0,'history':v}) 
for t in range(0,int(tmax/dt)+1):
    vn = v.copy() # Previous values
    rhou = vn[0,:]*vn[1,:]
    rhouu = vn[0,:]*vn[1,:]*vn[1,:]
    u = vn[1,:]/vn[0,:]
    T = 1/(Cp-R) * (vn[2,:]/vn[0,:] - u*u/2)
    P = vn[0,:] * R * T  # rho R T
    Cp = air.cp(T=T,p=P/101325) * 1000
    mu = sutherland(T)
    lam = -2/3 * mu 
    Re = vn[0,:] * u * Ls / mu  # Reynolds number based on reference length 

    # Left boundary Upwind
    # Continunity
    i=0
    drhou_dx = -(vn[1,i+1] - vn[1,i])/(dx)
    v[0,i] = dt*drhou_dx + vn[0,i] # rho @ t+1
    
    # Momentum
    drhouu_dx = (rhouu[i+1] - rhouu[i])/(dx)
    a =  drhouu_dx + P0_s / (rho_s*u_s*u_s) * (P[i+1]- P[i])/dx - (2 + lam[i]/mu[i]) / Re_s * (u[i+1] - 2*u[i] + u[i-1])/(dx*dx)
    v[1,i] = dt * (-a) + vn[1,i]

    # Energy
    drho_u_Es_dx = ( vn[1,i+1]*vn[2,i+1] - vn[1,i]*vn[2,i] )/(dx) 
    dP_u_dx = (P[i+1]*u[i+1] - P[i]*u[i]) / (dx)
    du_dx = (u[i+1] - u[i])/(dx)
    ddu_dxx = (u[i+2] - 2*u[i+1] + u[i] ) / (dx*dx)
    a = rho_s*u_s*E_s / Ls * drho_u_Es_dx + P0_s * u_s / Ls * (dP_u_dx) \
        - u_s*u_s / Ls * mu[i] * (lam[i] / mu[i] + 2) * ( du_dx*du_dx + u[i] * ddu_dxx)
    v[2,i] = dt*Ls/(E_s*u_s)* (-a) + vn[2,i]

    # Central Differences 
    for i in range(1,len(x)-1):
        # Continunity
        drhou_dx = -(vn[1,i+1] - vn[1,i-1])/(2*dx)
        v[0,i] = dt*drhou_dx + vn[0,i] # rho @ t+1
        # Momentum
        drhouu_dx = (rhouu[i+1] - rhouu[i-1])/(2*dx)
        a =  drhouu_dx +  P0_s / (rho_s*u_s*u_s) * (P[i+1]-P[i-1])/(2*dx) - (2 + lam[i]/mu[i]) / Re_s * (u[i+1] - 2*u[i] + u[i-1])/(dx*dx)
        v[1,i] = dt * (-a) + vn[1,i]
        # if np.isnan(v[1,i]):
        #     print("check")
        # Energy
        drho_u_Es_dx = ( vn[1,i+1]*vn[2,i+1] - vn[1,i-1]*vn[2,i-1] )/(2*dx) 
        dP_u_dx = (P[i+1]*u[i+1] - P[i-1]*u[i-1]) / (2*dx)
        du_dx = (u[i+1] - u[i-1])/(2*dx)
        ddu_dxx = (u[i+1] - 2*u[i] + u[i-1] ) / (dx*dx)
        a = rho_s*u_s*E_s / Ls * drho_u_Es_dx + P0_s * u_s / Ls * (dP_u_dx) \
            - u_s*u_s / Ls * mu[i] * (lam[i] / mu[i] + 2) * ( du_dx*du_dx + u[i] * ddu_dxx)
        v[2,i] = dt*Ls/(E_s*u_s)* (-a) + vn[2,i]

    # Right boundary upwind
    # Continunity
    i=len(x)-1
    drhou_dx = -(vn[1,i] - vn[1,i-1])/(dx)
    v[0,i] = dt*drhou_dx + vn[0,i] # rho @ t+1
    
    # Momentum
    drhouu_dx = (rhouu[i] - rhouu[i-1])/(dx)
    a =  drhouu_dx +  P0_s / (rho_s*u_s*u_s) * (P[i]- P[i-1])/dx - (2 + lam[i]/mu[i]) / Re_s * (u[i] - 2*u[i-1] + u[i-2])/(dx*dx)
    v[1,i] = dt * (-a) + vn[1,i]

    # Energy
    drho_u_Es_dx = ( vn[1,i]*vn[2,i] - vn[1,i-1]*vn[2,i-1] )/(dx) 
    dP_u_dx = (P[i]*u[i] - P[i-1]*u[i-1]) / (dx)
    du_dx = (u[i] - u[i-1])/(dx)
    ddu_dxx = (u[i] - 2*u[i-1] + u[i-2] ) / (dx*dx)
    a = rho_s*u_s*E_s / Ls * drho_u_Es_dx + P0_s * u_s / Ls * (dP_u_dx) \
        - u_s*u_s / Ls * mu[i] * (lam[i] / mu[i] + 2) * ( du_dx*du_dx + u[i] * ddu_dxx)
    v[2,i] = dt*Ls/(E_s*u_s)* (-a) + vn[2,i]

    # Statistics
    u = v[1,:]/v[0,:]
    T = 1/(Cp-R) * (v[2,:]/v[0,:] - u*u/2)
    P = v[0,:] * R * T  # rho R T
    Cp = air.cp(T=T,p=P/101325) * 1000

    u = v[1,:]/v[0,:]
    umax = max(u)*u_s
    uavg = np.mean(u)*u_s
    # print(f"t={t*dt} max velocity {umax}, average velocity {uavg}")
    # Save the results
    history.append({'t':t*dt, 'v':v,'Cp':Cp,'P':P,'T':T,'u':u})

# Write to pickle file 
settings['history'] = history

with open('history.pickle', 'wb') as handle:
    pickle.dump(settings, handle, protocol=pickle.HIGHEST_PROTOCOL)

