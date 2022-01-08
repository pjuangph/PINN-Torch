import sys, torch
from typing import Dict
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0,'../../nangs')
from burgers_lib import initialize, create_domain, plot_domain_2D, plot_history_2D
from ml.burgers_pde import burgers_pde
from nangs import Dirichlet, MLP 
from nangs.samplers import Custom_Sampler,RandomSampler
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
    nx,ny = X.shape[0],X.shape[1]
    
    u_bounds = settings['u_bounds']
    v_bounds = settings['v_bounds']
    """
        Initialization:
        - The inputs are the denominators of the physics equation. 
        - Outputs are the quantities of interest. For the Burger's equation we want 
    """
    def initial_conditions_func(inputs):
        Lx = settings['x']['max'] -  settings['x']['min']
        Ly = settings['y']['max'] -  settings['y']['min']
        
        # U 
        x_lb = settings['x']['min'] + u_bounds['x_percent'][0]*Lx
        x_ub = settings['x']['min'] + u_bounds['x_percent'][1]*Lx

        y_lb = settings['y']['min'] + u_bounds['y_percent'][0]*Ly
        y_ub = settings['y']['min'] + u_bounds['y_percent'][1]*Ly

        comparison = torch.logical_and(torch.logical_and(inputs['x']>=x_lb, inputs['x']<= x_ub), torch.logical_and(inputs['y']>=y_lb, inputs['y']<=y_ub))
        
        u = comparison * settings['u']['max'] + (~comparison) * settings['u']['min']
        
        # V
        x_lb = settings['x']['min'] + v_bounds['x_percent'][0]*Lx
        x_ub = settings['x']['min'] + v_bounds['x_percent'][1]*Lx

        y_lb = settings['y']['min'] + v_bounds['y_percent'][0]*Ly
        y_ub = settings['y']['min'] + v_bounds['y_percent'][1]*Ly

        comparison = torch.logical_and(torch.logical_and(inputs['x']>=x_lb, inputs['x']<= x_ub), torch.logical_and(inputs['y']>=y_lb, inputs['y']<=y_ub))

        v = comparison * settings['v']['max'] + (~comparison) * settings['u']['min']

        return {'u': u.type(torch.FloatTensor).to(device),
                'v':v.type(torch.FloatTensor).to(device)}


    initial_conditions = Dirichlet(
        RandomSampler({'x': [0., 2.], 'y': [0., 2.], 't': 0.}, device=device, n_samples=1000), 
        lambda inputs: initial_conditions_func(inputs),
        name="initial-conditions"
    )

    pde = burgers_pde(inputs=('x', 'y', 't'), outputs=('u', 'v'),nu=settings['nu'])

    """
        Set the wall boundary condition
    """
    wall_bottom = Dirichlet(
        RandomSampler({'x': [0., 2.], 'y': 0., 't': [0., settings['tmax']]}, device=device, n_samples=1000), 
        lambda inputs: {'u': torch.as_tensor(1*inputs['x']).to(device), 'v': torch.as_tensor(1*inputs['x']).to(device)},
        name="wall-bottom"
    )

    wall_top = Dirichlet(
        RandomSampler({'x': [0., 2.], 'y': 2., 't': [0., settings['tmax']]}, device=device, n_samples=1000), 
        lambda inputs: {'u': torch.as_tensor(1*inputs['x']).to(device), 'v': torch.as_tensor(1*inputs['x']).to(device)},
        name="wall-top"
    )

    wall_left = Dirichlet(
        RandomSampler({'x': 0., 'y': [0, 2.], 't': [0, settings['tmax']]}, device=device, n_samples=1000), 
        lambda inputs: {'u': torch.as_tensor(1*inputs['x']).to(device), 'v': torch.as_tensor(1*inputs['x']).to(device)},
        name="wall-left"
    )

    wall_right = Dirichlet(
        RandomSampler({'x': 2., 'y': [0, 2.], 't': [0, settings['tmax']]}, device=device, n_samples=1000), 
        lambda inputs: {'u': torch.as_tensor(1*inputs['x']).to(device), 'v': torch.as_tensor(1*inputs['x']).to(device)},
        name="wall-right"
    )

    """
        Solving the PDE
    """
    pde_sampler = RandomSampler({
        'x': [0.,2.],
        'y': [0.,2.],
        't': [0., settings['tmax']]
    },device=device, n_samples=20000)

    pde.set_sampler(pde_sampler)
    pde.add_boco(initial_conditions)
    pde.add_boco(wall_bottom)
    pde.add_boco(wall_top)
    pde.add_boco(wall_left)
    pde.add_boco(wall_right)

    # solve
    n_inputs = len(pde.inputs)
    n_outputs = len(pde.outputs)
    hidden_layers = [64,64,64,64]
    n_steps = 5000
    mlp = MLP(n_inputs,n_outputs,5,128).to(device) # MultiLayerLinear(n_inputs, n_outputs, hidden_layers).to(device)
    optimizer = torch.optim.Adam(mlp.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, pct_start=0.1, div_factor=2, final_div_factor=1, total_steps=n_steps)

    pde.compile(mlp, optimizer, scheduler)
    hist = pde.solve(n_steps)

    # save the model
    torch.save({'model':mlp.state_dict(),
                'optimizer':optimizer.state_dict(),
                'history':hist,
                'num_inputs':n_inputs,
                'num_outputs':n_outputs,
                'hidden_layers':hidden_layers,
                'tmax':settings['tmax'],
                }, 'burgers_train.pt')

    
    # 'scalers': {'x_scaler':x_scaler,'y_scaler':y_scaler,'u_scaler':u_scaler,'v_scaler':v_scaler}
    
    # plot_domain_2D('burgers_2D',x,y,u_history[-1],v_history[-1])
    # plot_history_2D('burgers_2D.gif',u_history,v_history,x,y)