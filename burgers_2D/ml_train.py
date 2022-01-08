import sys, torch
from typing import Dict

from ml.MultiLayerLinear import MultiLayerLinear
sys.path.insert(0,'../../nangs')
from burgers_lib import initialize, create_domain, plot_domain_2D, plot_history_2D
from ml.burgers_pde import burgers_pde
from nangs import Dirichlet, MLP 
from nangs.samplers import Custom_Sampler
import numpy as np 
import json 

device = "cuda" if torch.cuda.is_available() else "cpu"

'''
    Burgers2D from same mesh points as the analytical model
'''
with open('settings.json','r') as f:
    settings = json.load(f)
    settings = settings['Burgers2D']
    x,y,u,v = create_domain(nx=settings['nx'],ny=settings['ny'])
    X,Y = np.meshgrid(x,y)
    t = np.arange(0,settings['tmax'],1.0) # settings['tmax'] = 0.001 is too agressive
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
        Custom_Sampler({'x': X.flatten(), 'y': Y.flatten(), 't': X.flatten()*0}, device=device, n_samples=len(X.flatten())), 
        lambda inputs: {'u': torch.as_tensor(u_init.flatten(),dtype=torch.float32,device=device), 'v': torch.as_tensor(v_init.flatten(),dtype=torch.float32,device=device)},
        name="initial"
    )
    
    # Enforce a strict wall boundary condition. All the x,y point values along the walls for all time will be equal to 1 in both u and v
    x_wall = np.concatenate((X[:,0], X[:,-1], X[0,:], X[-1,:]),axis=0)
    y_wall = np.concatenate([Y[:,0], Y[:,-1], Y[0,:], Y[-1,:]],axis=0)
    n_wall = len(x_wall)
    x_wall = x_wall.repeat(len(t),axis=0)
    y_wall = y_wall.repeat(len(t),axis=0)
    t_wall = np.concatenate([np.zeros((n_wall,1))+t[i] for i in range(len(t))])[:,0]

    x_wall = torch.as_tensor(x_wall,dtype=torch.float32,device=device)
    y_wall = torch.as_tensor(y_wall,dtype=torch.float32,device=device)
    t_wall = torch.as_tensor(t_wall,dtype=torch.float32,device=device)
    u_wall = torch.as_tensor(t_wall*0+1.0,dtype=torch.float32,device=device) # velocity in u is set to 1
    v_wall = torch.as_tensor(t_wall*0+1.0,dtype=torch.float32,device=device)

    wall_boundary = Dirichlet(
        Custom_Sampler({'x': x_wall, 'y': y_wall, 't': t_wall}, device=device, n_samples=len(x_wall)), 
        lambda inputs: {'u' : u_wall, 'v': v_wall},
        name="wall"
    ) # lambda inputs is typically a function but in this case, it returns a simple dictionary object

    """
        Solving the PDE
    """
    # You can also create a 3-Dimensional. So you have X.shape (nx,ny) there's an x for every y and a y for every x the 3rd dimension is time.
    # Take the 3D array and flatten it.
    sampler = Custom_Sampler({
        'x': X.flatten(), 
        'y': Y.flatten(),
        't': torch.as_tensor(np.random.random_sample(len(X.flatten())) * settings['tmax'],device=device,dtype=torch.float32)
    }, device=device, n_samples=1000)           # Here we are solving the entire equation for all times 1000 samples

    pde.set_sampler(sampler)
    pde.add_boco(initial_condition)
    pde.add_boco(wall_boundary)

    # solve
    n_inputs = len(pde.inputs)
    n_outputs = len(pde.outputs)
    hidden_layers = [64,64,64,64]
    n_steps = 5000
    mlp = MultiLayerLinear(n_inputs, n_outputs, hidden_layers).to(device)
    optimizer = torch.optim.Adam(mlp.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, pct_start=0.1, div_factor=10, final_div_factor=1, total_steps=n_steps)

    pde.compile(mlp, optimizer, scheduler)
    hist = pde.solve(n_steps)

    # save the model
    torch.save({'model':mlp.state_dict(),
                'optimizer':optimizer.state_dict(),
                'history':hist,
                'num_inputs':n_inputs,
                'num_outputs':n_outputs,
                'hidden_layers':hidden_layers,
                }, 'burgers_train.pt')
    # plot_domain_2D('burgers_2D',x,y,u_history[-1],v_history[-1])
    # plot_history_2D('burgers_2D.gif',u_history,v_history,x,y)