#import os, sys
import numpy as np
import json 
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', family='serif')
rc('lines', linewidth=1.5)
rc('font', size=14)
#plt.rc('legend',**{'fontsize':11})

def flux_ausm(q:np.ndarray,gamma:float):
    """AUSM Flux Splitting Scheme
    
    References:
        Liou, M. S., & Steffen Jr, C. J. (1993). A new flux splitting scheme. Journal of Computational physics, 107(1), 23-39.
        https://github.com/PauloBarrosCorreia/AUSM/blob/master/ausm.f90 

    Args:
        q (np.ndarray): [3,nx] [rho, rhou, rhoE]
        gamma (float): ratio of specific heats 

    Returns:
        np.ndarray: Flux
    """
    # AUSM Mach Number 
    r = q[0,:]
    u = q[1,:]/r
    E = q[2,:]/r
    P=(gamma-1.)*r*(E-0.5*u**2)
    a = np.sqrt(gamma*P/r)      
    M = u/a                     # Computes mach number at each location on the grid 

    M_plus = np.less_equal(np.abs(M),1)*0.25*(M+1)**2 + \
                np.greater(np.abs(M),1) * 0.5*(M+abs(M)) # M+, np.less_than/np.greater produces a boolean array [0,0,0,1,1,1] etc 

    M_neg = np.less_equal(np.abs(M),1)*(-0.25*(M-1)**2) + \
                np.greater(np.abs(M),1)*0.5*(M-abs(M)) # M- 
    
    M_half = M_plus[1:] + M_neg[0:-1]  # (M+ index 1 to end) + (M- index 0 to end-1)

    # AUSM Pressure

    P_plus = np.less_equal(M,1)*0.25*P*(M+1)*(M+1) * (2.0 - M) \
                + np.greater(M,1)*0.5*P*(M+np.abs(M)) / M

    P_neg = np.less_equal(M,1)*0.25*P*(M-1)*(M-1) * (2.0 + M) \
                + np.greater(M,1)*0.5*P*(M-np.abs(M)) / M
        
    # P_half = P_plus[1,:] + P_neg[0:-1]
    
    H = E+P/r
    F_L = np.stack([r*a, r*a*u, r*a*H])[:,0:-1]
    F_R = np.stack([r*a, r*a*u, r*a*H])[:,1:]
    
    # AUSM Discretization
    F_half = M_half * 0.5 * ( F_L + F_R ) - 0.5 * np.abs(M_half) * (F_R-F_L) + P_plus[0:-1] + P_neg[1:] # Equation 9 from AUSM Paper 

    return F_half # F

def update_euler(q:np.ndarray,F:np.ndarray,lamda:float,ng:int) -> np.ndarray:
    """Updates the solution

    Args:
        q (np.ndarray): [3,nx] [rho, rhou, rhoE]
        f (np.ndarray): _description_
        lambda (float): dt/dx
        ng (int): number of ghost cells

    Returns:
        np.ndarray: qnew, updated q vector 
    """
    qnew = q.copy()
    for i in range(q.shape[1]):
        qnew[i,:] = lamda * (F[:,i+1]-F[:,i-1]) + q[i,:]
        
    # Update ghost cells 
    nx = qnew.shape[1]
    qnew[:,0:ng] = q[:,0:ng] # set ghost cell qnew back to initial conditions
    qnew[:,(nx-ng):] = q[:,(nx-ng):] # set ghost cell qnew back to initial conditions
    return qnew

def update_RK4(q:np.ndarray,dt:float,dx:float,ng:int,gamma:float=1.4) -> np.ndarray:
    """Updates the solution using Runga-Kutta 4th order

    Args:
        q (np.ndarray): each row consists of [rho, rhou, rhoE]        
        dt (float): delta time
        dx (float): delta x coordinate
        ng (int): Number of points in x direction

    Returns:
        np.ndarray: qnew
    """
    qnew = q.copy()
    # q(t+dt) = G(U)
    # G(U) = (F(U)_(i+1) - F(U)_(i-1)) / dx
    k = qnew.copy()*0
            # Central Differencing
    k1 = (flux_ausm(q[2:],gamma,nx) - flux_ausm(q[0:-2],gamma,nx)) / (2*dx) 
    k2 =  (flux_ausm(q[2:]+k1*dt/2,gamma,nx) - flux_ausm(q[0:-2]+k1*dt/2,gamma,nx)) / (2*dx) 
    k3 =  (flux_ausm(q[2:]+k2*dt/2,gamma,nx) - flux_ausm(q[0:-2]+k2*dt/2,gamma,nx)) / (2*dx) 
    k4 =  (flux_ausm(q[2:]+k3*dt,gamma,nx) - flux_ausm(q[0:-2]+k3*dt,gamma,nx)) / (2*dx) 
    avg_slope = 1/6 *(k1+2*k2+2*k3+k4)
    qnew = k1 + avg_slope


    # Update ghost cells 
    nx = qnew.shape[1]
    qnew[:,0:ng] = q[:,0:ng] # set ghost cell qnew back to initial conditions
    qnew[:,(nx-ng):] = q[:,(nx-ng):] # set ghost cell qnew back to initial conditions
    return qnew

with open('settings.json','r') as f:
    settings = json.load(f)
    config = [c for c in settings['Configurations'] if c['id'] == settings['Configuration_to_run']][0]

# Parameters
CFL    = config['CFL']               # Courant Number
gamma  = config['gamma']             # Ratio of specific heats
ncells = settings['ncells']          # Number of cells
x_ini =0.; x_fin = 1.       # Limits of computational domain
dx = (x_fin-x_ini)/ncells   # Step size
nx = ncells+1               # Number of points
nghost_cells = 1           # Number of ghost cells on each boundary
x = np.arange(x_ini-dx*nghost_cells+dx/2, x_fin+dx*nghost_cells+dx/2, dx) # Mesh

# Build IC
r0 = np.zeros(nx)
u0 = np.zeros(nx)
p0 = np.zeros(nx)
halfcells = int(ncells/2)


p0[:halfcells] = config['left']['p0']; p0[halfcells:] = config['right']['p0'] 
u0[:halfcells] = config['left']['u0']; u0[halfcells:] = config['right']['u0']
r0[:halfcells] = config['left']['r0']; r0[halfcells:] = config['right']['r0']
tEnd = settings['tmax']

E0 = p0/((gamma-1.)*r0)+0.5*u0**2 # Total Energy density
a0 = np.sqrt(gamma*p0/r0)            # Speed of sound
q  = np.array([r0,r0*u0,r0*E0])   # Vector of conserved variables

# Solver loop
t  = 0
it = 0
a  = a0
dt=CFL*dx/max(abs(u0)+a0)         # Using the system's largest eigenvalue

while t < tEnd:
    q0 = q.copy()
    dF = flux_ausm(q0,dx,gamma,a,nx)
    
    q[:,1:-2] = q0[:,1:-2]-dt/dx*dF
    q[:,0]=q0[:,0]; q[:,-1]=q0[:,-1] # Dirichlet BCs
    
    # Compute primary variables
    rho=q[0]
    u=q[1]/rho
    E=q[2]/rho
    p=(gamma-1.)*rho*(E-0.5*u**2)
    a=np.sqrt(gamma*p/rho)
    if min(p)<0: print ('negative pressure found!')
    
    # Update/correct time step
    dt=CFL*dx/max(abs(u)+a)
    
    update_euler(q)
    # Update time and iteration counter
    t=t+dt; it=it+1
      
    # # Plot solution
    # if it%2 == 0:
    #     print (it)
    #     fig,axes = plt.subplots(nrows=4, ncols=1)
    #     plt.subplot(4, 1, 1)
    #     #plt.title('Roe scheme')
    #     plt.plot(x, rho, 'k-')
    #     plt.ylabel('$rho$',fontsize=16)
    #     plt.tick_params(axis='x',bottom=False,labelbottom=False)
    #     plt.grid(True)

    #     plt.subplot(4, 1, 2)
    #     plt.plot(x, u, 'r-')
    #     plt.ylabel('$U$',fontsize=16)
    #     plt.tick_params(axis='x',bottom=False,labelbottom=False)
    #     plt.grid(True)

    #     plt.subplot(4, 1, 3)
    #     plt.plot(x, p, 'b-')
    #     plt.ylabel('$p$',fontsize=16)
    #     plt.tick_params(axis='x',bottom=False,labelbottom=False)
    #     plt.grid(True)
    
    #     plt.subplot(4, 1, 4)
    #     plt.plot(x, E, 'g-')
    #     plt.ylabel('$E$',fontsize=16)
    #     plt.grid(True)
    #     plt.xlim(x_ini,x_fin)
    #     plt.xlabel('x',fontsize=16)
    #     plt.subplots_adjust(left=0.2)
    #     plt.subplots_adjust(bottom=0.15)
    #     plt.subplots_adjust(top=0.95)
    #     #plt.show()
    #     fig.savefig("analytical_plots/fig_Sod_Roe_it"+str(it)+".png", dpi=300)
