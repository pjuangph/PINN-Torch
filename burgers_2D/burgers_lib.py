from os import mkdir
from typing import Dict, List, Tuple
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation, rc
from pathlib import Path

def create_domain(nx:int=80, ny:int=80,xmax:float=2,ymax:float=2) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Creates the domain

    Args:
        nx (int, optional): Number of points in the x direction. Defaults to 80.
        ny (int, optional): Number of points in the y direction. Defaults to 80.
    
    Returns:
        (tuple): containing:

            **x** (np.ndarray): domain x direction
            **y** (np.ndarray): domain y direction
            **u** (np.ndarray): matrix describing x-velocity field in i,j space
            **v** (np.ndarray): matrix describing y-velocity field in i,j space
    """

    x = np.linspace(0,xmax,nx)
    y = np.linspace(0,ymax,ny)
    x,y = np.meshgrid(x,y)
    u = x.copy()
    v = x.copy()

    return x,y,u,v

def initialize_uv(x:np.ndarray,y:np.ndarray,u:np.ndarray,v:np.ndarray,u_max:float,v_max:float,u_min:float,v_min:float,u_bounds:Dict[str,float], v_bounds:Dict[str,float]) -> Tuple[np.ndarray,np.ndarray]:
    """initializes u and v 

    Args:
        u (np.ndarray): matrix containing u. u[i,j] with shape nx, ny 
        v (np.ndarray): matrix containing v. v[i,j] with shape nx, ny 
        u_max (float): max value to initialize u with inside u_bounds
        v_max (float): max value to initialize v with inside v_bounds
        u_min (float): min value to initialize u with outside u_bounds
        v_min (float): min value to initialize v with outside v_bounds
        u_bounds (Dict[str,float]): Defines the percentage of the domain to set to u_max for example u_bounds = [(0.1,0.9),(0.2,0.2)] equates to 10% to 90% of the x direction and 20% to 80% in j direction to set value of u to be u_max
        v_bounds (Dict[str,float]): Defines the percentage of the domain to set to u_max for example v_bounds = [(0.1,0.9),(0.2,0.2)] equates to 10% to 90% of the y direction and 20% to 80% in j direction to set value of u to be v_max

    Returns:
        (tuple): containing:

            **u** (np.ndarray): Initialized velocity field in x direction
            **v** (np.ndarray): Initialized velocity field in y direction 
    """ 
    u[:,:] = u_min
    v[:,:] = v_min    
    Lx = x.max() -  x.min()
    Ly = y.max() -  y.min()
    x_lb = x.min() + u_bounds['x_percent'][0]*Lx
    x_ub = x.min() + u_bounds['x_percent'][1]*Lx

    y_lb = y.min() + u_bounds['y_percent'][0]*Ly
    y_ub = y.min() + u_bounds['y_percent'][1]*Ly

    comparison = np.logical_and(np.logical_and(x>=x_lb, x<= x_ub), np.logical_and(y>=y_lb, y<=y_ub))
    u = comparison * u_max + (~comparison) * u_min
    
    # V
    x_lb = x.min() + v_bounds['x_percent'][0]*Lx
    x_ub = x.min() + v_bounds['x_percent'][1]*Lx

    y_lb = y.min() + v_bounds['y_percent'][0]*Ly
    y_ub = y.min() + v_bounds['y_percent'][1]*Ly

    comparison = np.logical_and(np.logical_and(x>=x_lb, x<= x_ub), np.logical_and(y>=y_lb, y<=y_ub))
    v = comparison * v_max + (~comparison) * v_min
    
    return u.astype(np.float64),v.astype(np.float64)


def plot_domain_2D(prefix:str,x:np.ndarray,y:np.ndarray,u:np.ndarray,v:np.ndarray,i:int):
    """Saves an image containing the value of u and v for burger's equation

    Args:
        prefix (str): filename prefix
        x (np.ndarray): matrix x direction
        y (np.ndarray): matrix y direction
        u (np.ndarray): matrix containing u. u[i,j] with shape nx, ny 
        v (np.ndarray): matrix containing v. v[i,j] with shape nx, ny 
    """    

    fig = plt.figure(figsize=(30,10), dpi=300,num=1,clear=True)
    plt.rcParams.update({'font.size': 16})
    ax1 = fig.add_subplot(121,projection="3d") # Plot of u
    ax1.plot_surface(x, y, u, cmap=cm.jet)
    ax1.xaxis.labelpad=18.0
    ax1.yaxis.labelpad=18.0
    ax1.set_xlabel('x direction')
    ax1.set_ylabel('y direction')
    ax1.set_zlabel('Velocity')
    ax1.set_title(f'{prefix} U - velocity')
    ax1.set_zlim([1,3.2])

    ax2 = fig.add_subplot(122,projection="3d") # Plot of v``    
    ax2.plot_surface(x, y, v, cmap=cm.jet)
    ax2.xaxis.labelpad=18.0
    ax2.yaxis.labelpad=18.0
    ax2.set_xlabel('x direction')
    ax2.set_ylabel('y direction')
    ax2.set_zlabel('Velocity')
    ax2.set_title(f'{prefix} V - velocity')
    ax2.set_zlim([1,3.2])
    # plt.show()
    Path("figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(f'figures/{prefix}_burgers_{i:010}.png')

def plot_history_2D(filename:str,u_history:List[np.ndarray],v_history:List[np.ndarray],x:np.ndarray,y:np.ndarray):
    """[summary]

    Args:
        filename (str): [description]
        u_history (List[np.ndarray]): [description]
        v_history (List[np.ndarray]): [description]
        x (np.ndarray): [description]
        y (np.ndarray): [description]

    Returns:
        [type]: [description]
    """
    num_frames=300
    divisor = num_frames/len(u_history)

    X, Y = np.meshgrid(x, y)
    u = u_history[0]
    v = u_history[0]
    fig = plt.figure(figsize=(15,5), dpi=300)
    ax1 = fig.add_subplot(121,projection="3d") # Plot of u
    ax2 = fig.add_subplot(122,projection="3d") # Plot of u
    
    data1 = ax1.plot_surface(X, Y, u, cmap=cm.jet)
    ax1.set_xlabel('x direction')
    ax1.set_ylabel('y direction')
    ax1.set_title('U - velocity')
    ax1.set_xlim(min(x),max(x))
    ax1.set_ylim(min(y),max(y)) 

    data2 = ax2.plot_surface(X, Y, v, cmap=cm.jet)
    ax2.set_xlabel('x direction')
    ax2.set_ylabel('y direction')
    ax2.set_title('V - velocity')
    ax2.set_xlim(min(x),max(x))
    ax2.set_ylim(min(y),max(y)) 

    def animate(i):
        i = int(i/divisor)
        u = u_history[i]
        v = u_history[i]
        ax1.plot_surface(X, Y, u, cmap=cm.jet)
        ax2.plot_surface(X, Y, v, cmap=cm.jet)
        # data1.set_data(X,Y,u)
        # data2.set_data(X,Y,v)
        # return (data1,data2)
    

    anim = animation.FuncAnimation(fig, animate, init_func=None,
                                frames=num_frames, interval=10, blit=True)
    # from IPython.display import HTML

    # HTML(anim.to_html5_video())