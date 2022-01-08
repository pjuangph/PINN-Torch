from typing import List, Tuple
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation, rc

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
    u = np.zeros(shape=(ny,nx))
    v = np.zeros(shape=(ny,ny))

    return x,y,u,v

def initialize(u:np.ndarray,v:np.ndarray,u_max:float,v_max:float,u_min:float,v_min:float,u_bounds:List[Tuple[float,float]], v_bounds:List[Tuple[float,float]]) -> Tuple[np.ndarray,np.ndarray]:
    """initializes u and v 

    Args:
        u (np.ndarray): matrix containing u. u[i,j] with shape nx, ny 
        v (np.ndarray): matrix containing v. v[i,j] with shape nx, ny 
        u_max (float): max value to initialize u with inside u_bounds
        v_max (float): max value to initialize v with inside v_bounds
        u_min (float): min value to initialize u with outside u_bounds
        v_min (float): min value to initialize v with outside v_bounds
        u_bounds (List[Tuple[float,float]]): Defines the percentage of the domain to set to u_max for example u_bounds = [(0.1,0.9),(0.2,0.2)] equates to 10% to 90% of the x direction and 20% to 80% in j direction to set value of u to be u_max
        v_bounds (List[Tuple[float,float]]): Defines the percentage of the domain to set to u_max for example v_bounds = [(0.1,0.9),(0.2,0.2)] equates to 10% to 90% of the y direction and 20% to 80% in j direction to set value of u to be v_max

    Returns:
        (tuple): containing:

            **u** (np.ndarray): Initialized velocity field in x direction
            **v** (np.ndarray): Initialized velocity field in y direction 
    """ 
    u[:,:] = u_min
    v[:,:] = v_min
    i = np.array(u_bounds[0])*u.shape[0]
    j = np.array(u_bounds[1])*u.shape[1]
    i = i.astype(int); j = j.astype(int)
    u[i[0]:i[1], j[0]:j[1]] = u_max

    i = np.array(v_bounds[0])*u.shape[0]
    j = np.array(v_bounds[1])*u.shape[1]
    i = i.astype(int); j = j.astype(int)
    v[i[0]:i[1], j[0]:j[1]] = v_max
    
    return u,v


def plot_domain_2D(prefix:str,x:np.ndarray,y:np.ndarray,u:np.ndarray,v:np.ndarray):
    """Saves an image containing the value of u and v for burger's equation

    Args:
        prefix (str): filename prefix
        x (np.ndarray): domain x direction
        y (np.ndarray): domain y direction
        u (np.ndarray): matrix containing u. u[i,j] with shape nx, ny 
        v (np.ndarray): matrix containing v. v[i,j] with shape nx, ny 
    """
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(30,10), dpi=300)
    ax1 = fig.add_subplot(121,projection="3d") # Plot of u
    ax1.plot_surface(X, Y, u, cmap=cm.jet)
    ax1.set_xlabel('x direction')
    ax1.set_ylabel('y direction')
    ax1.set_zlabel('Velocity')
    ax1.set_title('U - velocity')
    ax1.set_zlim([0,3])

    ax2 = fig.add_subplot(122,projection="3d") # Plot of v``    
    ax2.plot_surface(X, Y, v, cmap=cm.jet)
    ax2.set_xlabel('x direction')
    ax2.set_ylabel('y direction')
    ax2.set_zlabel('Velocity')
    ax2.set_title('V - velocity')
    ax2.set_zlim([0,3])
    plt.show()
    # plt.savefig(f'{prefix}_burgers_u_v.png')
    
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