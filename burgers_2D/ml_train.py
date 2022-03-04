import sys, torch
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
    def compute_initial_condition(inputs):
        Lx = settings['x']['max'] -  settings['x']['min']
        Ly = settings['y']['max'] -  settings['y']['min']

        # U 
        x_lb = settings['x']['min'] + u_bounds['x_percent'][0]*Lx
        x_ub = settings['x']['min'] + u_bounds['x_percent'][1]*Lx

        y_lb = settings['y']['min'] + u_bounds['y_percent'][0]*Ly
        y_ub = settings['y']['min'] + u_bounds['y_percent'][1]*Ly

        comparison = torch.logical_and(torch.logical_and(inputs['x']>=x_lb, inputs['x']<= x_ub), torch.logical_and(inputs['y']>=y_lb, inputs['y']<=y_ub))
        
        u = comparison * settings["u"]["max"] + (~comparison) * settings["u"]["min"]
        
        # V
        x_lb = settings['x']['min'] + v_bounds['x_percent'][0]*Lx
        x_ub = settings['x']['min'] + v_bounds['x_percent'][1]*Lx

        y_lb = settings['y']['min'] + v_bounds['y_percent'][0]*Ly
        y_ub = settings['y']['min'] + v_bounds['y_percent'][1]*Ly

        comparison = torch.logical_and(torch.logical_and(inputs['x']>=x_lb, inputs['x']<= x_ub), torch.logical_and(inputs['y']>=y_lb, inputs['y']<=y_ub))

        v = comparison * settings["v"]["max"] + (~comparison) * settings["v"]["min"]

        return {'u': u.type(torch.FloatTensor).to(device),
                'v': v.type(torch.FloatTensor).to(device)}

    # x = torch.linspace(0,2,50)
    # y = torch.linspace(0,2,50)
    # xx, yy = torch.meshgrid(x,y)

    # ic = compute_initial_condition({'x': xx, 'y': yy})
    # u,v= ic['u'].cpu().detach().numpy(), ic['v'].cpu().detach().numpy()

    # fig = plt.figure(figsize=(30,10), dpi=300)
    # ax1 = fig.add_subplot(121,projection="3d") # Plot of u
    # ax1.plot_surface(xx,yy, u, cmap=cm.jet)
    # ax1.set_xlabel('x direction')
    # ax1.set_ylabel('y direction')
    # ax1.set_zlabel('Velocity')
    # ax1.set_title('U - velocity')
    # # ax1.set_zlim([-1,1])

    # ax2 = fig.add_subplot(122,projection="3d") # Plot of v``    
    # ax2.plot_surface(xx, yy, v, cmap=cm.jet)
    # ax2.set_xlabel('x direction')
    # ax2.set_ylabel('y direction')
    # ax2.set_zlabel('Velocity')
    # ax2.set_title('V - velocity')
    # plt.show()

    initial_conditions = Dirichlet(
        RandomSampler({'x': [settings['x']['min'], settings['x']['max']], 'y': [settings['y']['min'], settings['y']['max']], 't': [-0.001,0]}, device=device, n_samples=1000), 
        compute_initial_condition,
        name="ics"
    )

    pde = burgers_pde(inputs=('x', 'y', 't'), outputs=('u', 'v'),nu=settings['nu'])

    """
        Set the wall boundary condition
    """
    wall_bottom = Dirichlet(
        RandomSampler({'x': [settings['x']['min'], settings['x']['max']], 'y': settings['y']['min'], 't': [0, settings['tmax']]}, device=device, n_samples=1000), 
        lambda inputs: {'u': torch.as_tensor(settings["u"]["min"]+0*inputs['x']).to(device), 'v': torch.as_tensor(settings["v"]["min"]+0*inputs['x']).to(device)},
        name="wb"
    ) # wall bottom

    wall_top = Dirichlet(
        RandomSampler({'x': [settings['x']['min'], settings['x']['max']], 'y': settings['y']['max'], 't': [0, settings['tmax']]}, device=device, n_samples=1000), 
        lambda inputs: {'u': torch.as_tensor(settings["u"]["min"]+0*inputs['x']).to(device), 'v': torch.as_tensor(settings["v"]["min"]+0*inputs['x']).to(device)},
        name="wt"
    ) # wall-top

    wall_left = Dirichlet(
        RandomSampler({'x': settings['x']['min'], 'y': [settings['y']['min'], settings['y']['max']], 't': [0, settings['tmax']]}, device=device, n_samples=1000), 
        lambda inputs: {'u': torch.as_tensor(settings["u"]["min"]+0*inputs['x']).to(device), 'v': torch.as_tensor(settings["v"]["min"]+0*inputs['x']).to(device)},
        name="wl"
    ) # wall left

    wall_right = Dirichlet(
        RandomSampler({'x': settings['x']['max'], 'y': [settings['y']['min'], settings['y']['max']], 't': [0, settings['tmax']]}, device=device, n_samples=1000), 
        lambda inputs: {'u': torch.as_tensor(settings["u"]["min"]+0*inputs['x']).to(device), 'v': torch.as_tensor(settings["v"]["min"]+0*inputs['x']).to(device)},
        name="wr"
    ) # wall right

    """
        Solving the PDE
    """
    pde_sampler = RandomSampler({
        'x': [settings['x']['min'], settings['x']['max']],
        'y': [settings['y']['min'], settings['y']['max']],
        't': [0, settings['tmax']]
    },device=device, n_samples=1000)

    pde.set_sampler(pde_sampler)
    pde.add_boco(initial_conditions)
    pde.add_boco(wall_bottom)
    pde.add_boco(wall_top)
    pde.add_boco(wall_left)
    pde.add_boco(wall_right)

    # solve
    LR = 1e-3
    n_inputs = len(pde.inputs)
    n_outputs = len(pde.outputs)
    n_layers = 6
    neurons = 64
    n_steps = 25000

    mlp = MLP(n_inputs,n_outputs,n_layers,neurons).to(device) # MultiLayerLinear(n_inputs, n_outputs, hidden_layers).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.2*n_steps),int(0.4*n_steps),int(0.6*n_steps),int(0.8*n_steps)], gamma=0.1)

    pde.compile(mlp, optimizer, scheduler)
    hist = pde.solve(n_steps)

    # save the model
    torch.save({'model':mlp.state_dict(),
                'optimizer':optimizer.state_dict(),
                'history':hist,
                'num_inputs':n_inputs,
                'num_outputs':n_outputs,
                'n_layers':n_layers,
                'neurons':neurons,
                'tmax':settings['tmax'],
                }, 'burgers_train.pt')

    
    # 'scalers': {'x_scaler':x_scaler,'y_scaler':y_scaler,'u_scaler':u_scaler,'v_scaler':v_scaler}
    
    # plot_domain_2D('burgers_2D',x,y,u_history[-1],v_history[-1])
    # plot_history_2D('burgers_2D.gif',u_history,v_history,x,y)