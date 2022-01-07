import sys, torch
from typing import Dict
sys.path.insert(0,'../../nangs')
from burgers_lib import initialize, create_domain, plot_domain_2D, plot_history_2D
from ml.burgers_pde import burgers_pde
from nangs import Dirichlet, Custom_Sampler, MLP 
import numpy as np 
import json 

device = "cuda" if torch.cuda.is_available() else "cpu"

def wall_func(inputs:Dict[str,np.ndarray]):
    """This is the wall function

    Args:
        inputs (Dict[str,np.ndarray]): The direlect boundary condition will call this function with inputs specified as a dictionary {'x': val, 'y':val, 't':val }

    Returns:
        [type]: [description]
    """
    return inputs['x']*0 + 1 # return a constant value at the wall boundary

'''
    Burgers2D from same mesh points as the analytical model
'''
with open('settings.json','r') as f:
    settings = json.load(f)
    settings = settings['Burgers2D']
    x,y,u,v = create_domain(nx=settings['nx'],ny=settings['ny'])
    X,Y = np.meshgrid(x,y)
    t = np.arange(0,settings['tmax'],settings['dt'])
    u_bounds = settings['u_bounds']
    v_bounds = settings['v_bounds']

    u_init,v_init = initialize(u,v,settings['u_max'],settings['v_max'],
                [u_bounds['i_percent'],u_bounds['j_percent']],
                [v_bounds['i_percent'],v_bounds['j_percent']])
    
    """
        Initialization:
        - The inputs are the denominators of the physics equation. 
        - Outputs are the quantities of interest. For the Burger's equation we want 
    """


    pde = burgers_pde(inputs=('x', 'y', 't'), outputs=('u', 'v'))

    """
        Set the wall boundary condition
    """
    initial_condition = Dirichlet(
        Custom_Sampler({'x': x, 'y': y, 't': 0.}, device=device, n_samples=len(x)), 
        lambda inputs: {'u': u_init, 'v': v_init},
        name="initial"
    )
    
    x_wall = np.concatenate((X[:,0], X[:,-1], X[0,:], X[-1,:]),axis=0)
    y_wall = np.concatenate([Y[:,0], Y[:,-1], Y[0,:], Y[-1,:]],axis=0)

    wall_boundary = Dirichlet(
        Custom_Sampler({'x': x_wall, 'y': y_wall, 't': t}, device=device, n_samples=len(x_wall)), 
        lambda inputs: {'u' : 1.0, 'v': 1.0},
        name="wall"
    ) # lambda inputs is typically a function but in this case, it returns a simple dictionary object

    """
        Solving the PDE
    """
    pde.add_boco(initial_condition)
    pde.add_boco(wall_boundary)

    # solve
    LR = 1e-2
    N_STEPS = 5000
    NUM_LAYERS = 3
    NUM_HIDDEN = 128
    mlp = MLP(len(pde.inputs), len(pde.outputs), NUM_LAYERS, NUM_HIDDEN).to(device)
    optimizer = torch.optim.Adam(mlp.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, pct_start=0.1, div_factor=10, final_div_factor=1, total_steps=N_STEPS)

    pde.compile(mlp, optimizer, scheduler)
    hist = pde.solve(N_STEPS)


    # plot_domain_2D('burgers_2D',x,y,u_history[-1],v_history[-1])
    # plot_history_2D('burgers_2D.gif',u_history,v_history,x,y)