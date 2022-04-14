#import os, sys
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font', family='serif')
rc('lines', linewidth=1.5)
rc('font', size=14)
#plt.rc('legend',**{'fontsize':11})

######################################################################
#             Solving 1-D Euler system of equations
#                     using the Roe scheme
#
#             dq_i/dt + df_i/dx = 0, for x \in [a,b]
#
# This code solves the Sod's shock tube problem (IC=1)
#
# t=0                                 t=tEnd
# Density                             Density
#   ****************|                 *********\
#                   |                           \
#                   |                            \
#                   |                             ****|
#                   |                                 |
#                   |                                 ****|
#                   ***************                       ***********
#
# Domain cells (I{i}) reference:
#
#                |           |   u(i)    |           |
#                |  u(i-1)   |___________|           |
#                |___________|           |   u(i+1)  |
#                |           |           |___________|
#             ...|-----0-----|-----0-----|-----0-----|...
#                |    i-1    |     i     |    i+1    |
#                |-         +|-         +|-         +|
#              i-3/2       i-1/2       i+1/2       i+3/2
#
######################################################################

def flux_ausm(q:np.ndarray,dx:float,gamma:float,a:float,nx:int):
    """AUSM Flux Splitting Scheme
    
        Liou, M. S., & Steffen Jr, C. J. (1993). A new flux splitting scheme. Journal of Computational physics, 107(1), 23-39.

    Args:
        q (np.ndarray): [3,nx] [rho, rhou, E]
        dx (float): _description_
        gamma (float): _description_
        a (float): _description_
        nx (int): _description_

    Returns:
        _type_: _description_
    """
    # AUSM Mach Number 
    r = q[0,:]
    u = q[1,:]/r
    E=q[2,:]/r

    M = u/a 
    P=(gamma-1.)*r*(E-0.5*u**2)

    M_plus = np.less_equal(M,1)*0.25*(M+1)**2 + \
                np.greater(M,1) * 0.5*(M+abs(M)) # M+

    M_neg = np.less_equal(M,1)*(-0.25*(M-1)**2) + \
                np.greater(M,1)*0.5*(M-abs(M)) # M- 
    
    M_half = M_plus[1:] + M_neg[0:-1]  # (M+ index 1 to end) + (M- index 0 to end-1)

    # AUSM Pressure

    P_plus = np.less_equal(M,1)*0.25*P*(M+1)*(M+1) * (2.0 - M) \
                + np.greater(M,1)*0.5*P*(M+np.abs(M)) / M

    P_neg = np.less_equal(M,1)*0.25*P*(M-1)*(M-1) * (2.0 + M) \
                + np.greater(M,1)*0.5*P*(M-np.abs(M)) / M
        
    P_half = P_plus[1,:] + P_neg[0:-1]
    
    H = E+P/r
    F_L = np.stack([r*a, r*a*u, r*a*H])[:,0:-1]
    F_R = np.stack([r*a, r*a*u, r*a*H])[:,1:]
    
    # AUSM Discretization
    F_half = M_half * 0.5 * ( F_L + F_R ) - 0.5 * np.abs(M_half) * (F_R-F_L) + P_plus[0:-1] + P_neg[1:] # Equation 9 from AUSM Paper 

    return F_half

# Parameters
CFL    = 0.50               # Courant Number
gamma  = 1.4                # Ratio of specific heats
ncells = 400                # Number of cells
x_ini =0.; x_fin = 1.       # Limits of computational domain
dx = (x_fin-x_ini)/ncells   # Step size
nx = ncells+1               # Number of points
x = np.linspace(x_ini+dx/2.,x_fin,nx) # Mesh

# Build IC
r0 = zeros(nx)
u0 = zeros(nx)
p0 = zeros(nx)
halfcells = int(ncells/2)

IC = 1 # 6 IC cases are available
if IC == 1:
    print ("Configuration 1, Sod's Problem")
    p0[:halfcells] = 1.0  ; p0[halfcells:] = 0.1;
    u0[:halfcells] = 0.0  ; u0[halfcells:] = 0.0;
    r0[:halfcells] = 1.0  ; r0[halfcells:] = 0.125;
    tEnd = 0.20;
elif IC== 2:
    print ("Configuration 2, Left Expansion and right strong shock")
    p0[:halfcells] = 1000.; p0[halfcells:] = 0.1;
    u0[:halfcells] = 0.0  ; u0[halfcells:] = 0.0;
    r0[:halfcells] = 3.0  ; r0[halfcells:] = 0.2;
    tEnd = 0.01;
elif IC == 3:
    print ("Configuration 3, Right Expansion and left strong shock")
    p0[:halfcells] = 7.   ; p0[halfcells:] = 10.;
    u0[:halfcells] = 0.0  ; u0[halfcells:] = 0.0;
    r0[:halfcells] = 1.0  ; r0[halfcells:] = 1.0;
    tEnd = 0.10;
elif IC == 4:
    print ("Configuration 4, Shocktube problem of G.A. Sod, JCP 27:1, 1978")
    p0[:halfcells] = 1.0  ; p0[halfcells:] = 0.1;
    u0[:halfcells] = 0.75 ; u0[halfcells:] = 0.0;
    r0[:halfcells] = 1.0  ; r0[halfcells:] = 0.125;
    tEnd = 0.17;
elif IC == 5:
    print ("Configuration 5, Lax test case: M. Arora and P.L. Roe: JCP 132:3-11, 1997")
    p0[:halfcells] = 3.528; p0[halfcells:] = 0.571;
    u0[:halfcells] = 0.698; u0[halfcells:] = 0.0;
    r0[:halfcells] = 0.445; r0[halfcells:] = 0.5;
    tEnd = 0.15;
elif IC == 6:
    print ("Configuration 6, Mach = 3 test case: M. Arora and P.L. Roe: JCP 132:3-11, 1997")
    p0[:halfcells] = 10.33; p0[halfcells:] = 1.0;
    u0[:halfcells] = 0.92 ; u0[halfcells:] = 3.55;
    r0[:halfcells] = 3.857; r0[halfcells:] = 1.0;
    tEnd = 0.09;

E0 = p0/((gamma-1.)*r0)+0.5*u0**2 # Total Energy density
a0 = sqrt(gamma*p0/r0)            # Speed of sound
q  = np.array([r0,r0*u0,r0*E0])   # Vector of conserved variables

# Solver loop
t  = 0
it = 0
a  = a0
dt=CFL*dx/max(abs(u0)+a0)         # Using the system's largest eigenvalue

while t < tEnd:
    q0 = q.copy();
    dF = flux_ausm(q0,dx,gamma,a,nx)
    
    q[:,1:-2] = q0[:,1:-2]-dt/dx*dF;
    q[:,0]=q0[:,0]; q[:,-1]=q0[:,-1]; # Dirichlet BCs
    
    # Compute primary variables
    rho=q[0];
    u=q[1]/rho;
    E=q[2]/rho;
    p=(gamma-1.)*rho*(E-0.5*u**2);
    a=sqrt(gamma*p/rho);
    if min(p)<0: print ('negative pressure found!')
    
    # Update/correct time step
    dt=CFL*dx/max(abs(u)+a);
    
    # Update time and iteration counter
    t=t+dt; it=it+1;
        
    # Plot solution
    if it%2 == 0:
        print (it)
        fig,axes = plt.subplots(nrows=4, ncols=1)
        plt.subplot(4, 1, 1)
        #plt.title('Roe scheme')
        plt.plot(x, rho, 'k-')
        plt.ylabel('$rho$',fontsize=16)
        plt.tick_params(axis='x',bottom=False,labelbottom=False)
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(x, u, 'r-')
        plt.ylabel('$U$',fontsize=16)
        plt.tick_params(axis='x',bottom=False,labelbottom=False)
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(x, p, 'b-')
        plt.ylabel('$p$',fontsize=16)
        plt.tick_params(axis='x',bottom=False,labelbottom=False)
        plt.grid(True)
    
        plt.subplot(4, 1, 4)
        plt.plot(x, E, 'g-')
        plt.ylabel('$E$',fontsize=16)
        plt.grid(True)
        plt.xlim(x_ini,x_fin)
        plt.xlabel('x',fontsize=16)
        plt.subplots_adjust(left=0.2)
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(top=0.95)
        #plt.show()
        fig.savefig("analytical_plots/fig_Sod_Roe_it"+str(it)+".png", dpi=300)
