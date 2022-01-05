import numpy as np
import copy 

def burgers_timestep(x:np.ndarray,y:np.ndarray,u:np.ndarray,v:np.ndarray,nt:int,dt:float=0.001,nu:float=0.1):
    """Solving burgers equation using finite difference with euler time step (1st order approximation) 

    Args:
        x (np.ndarray): domain x direction
        y (np.ndarray): domain y direction
        u (np.ndarray): matrix describing x-velocity field in i,j space
        v (np.ndarray): matrix describing y-velocity field in i,j space
        nt (int): number of time advances
        dt (float): time step to use for first order approximation
        nu (float, Optional): Viscosity. Defaults to 0.1
    
    Returns:
        (tuple): containing

            **u_history** (np.ndarray): history for u velocity 
            **v_history** (np.ndarray): history for v velocity 
    """


    dx = x[1]-x[0] # We can do it this way because x and y are initialized using linspace which guarantees constant spacing 
    dy = y[1]-y[0]

    u_history = list()
    v_history = list() 
    u_history.append(copy.deepcopy(u)) # set equal to initial value
    v_history.append(copy.deepcopy(v))
    for n in range(nt):
        un = u.copy()   # previous value
        vn = v.copy()
        for i in range(1,len(x)-1):
            for j in range(1,len(y)-1):
                # Uses backward difference in space to solve first order derivative
                # Central differencing for second order derivative 
                u[i,j] = (un[i, j] -(un[i, j] * dt / dx * (un[i, j] - un[i-1, j])) -vn[i, j] * dt / dy * (un[i, j] - un[i, j-1])) + (nu*dt/(dx**2))*(un[i+1,j]-2*un[i,j]+un[i-1,j])+(nu*dt/(dx**2))*(un[i,j-1]-2*un[i,j]+un[i,j+1])
                v[i,j] = (vn[i, j] -(un[i, j] * dt / dx * (vn[i, j] - vn[i-1, j]))-vn[i, j] * dt / dy * (vn[i, j] - vn[i, j-1])) + (nu*dt/(dx**2))*(vn[i+1,j]-2*vn[i,j]+vn[i-1,j])+(nu*dt/(dx**2))*(vn[i,j-1]-2*vn[i,j]+vn[i,j+1])
        
        u[:,0] = 1       # At all i values when j = 0
        u[:,-1] = 1      # At all i values when j = jmax
        u[0,:] = 1       # At all j values and i = 0
        u[-1,:] = 1     # At all j values and i = imax

        v[:,0] = 1
        v[:,-1] = 1
        v[0,:] = 1
        v[-1,:] = 1

        u_history.append(copy.deepcopy(u))
        v_history.append(copy.deepcopy(v))
    return u_history, v_history
    
