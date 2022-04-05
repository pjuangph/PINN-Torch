import sys, torch
from typing import Dict
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0,'../../nangs')
from nangs import Dirichlet, MLP 
from nangs.samplers import RandomSampler
import numpy as np 
import json 
import matplotlib.pyplot as plt 
from matplotlib import cm
from shocktube_pde import shocktube_pde

device = "cuda" if torch.cuda.is_available() else "cpu"

'''
    ShockTube
'''
with open('settings.json','r') as f:
    settings = json.load(f)    
    config = [c for c in settings['Configurations'] if c['id'] == settings['Configuration_to_run']][0]
    ncells = settings['ncells']
    x_ini =0.0; x_fin = settings['xmax']       # Limits of computational domain
    half = (x_fin + x_ini)/2.0
    
    dx = (x_fin-x_ini)/ncells   # Step size
    nx = ncells+1               # Number of points
    x = np.linspace(x_ini+dx/2.,x_fin,nx) # Mesh

    def compute_initial_condition(inputs:Dict[str,float]):
        """Initialization

        Args:
            inputs (Dict[str,float]): Inputs are p0, u0, r0 

        Returns:
            Dict[str,float]: _description_
        """
        comparison = inputs['x']<half
        
        p0 = comparison * config['left']['p0'] + (~comparison) * config['right']['p0']
        u0 = comparison * config['left']['u0'] + (~comparison) * config['right']['u0']
        rho0 = comparison * config['left']['r0'] + (~comparison) * config['right']['r0']

        return {'p': p0,
                'u': u0,
                'rho': rho0}


    initial_conditions = Dirichlet(
        RandomSampler({ 't': 0.0,
                        'x': [0.0, settings['xmax']]},
                        device=device, n_samples=1000), 
        compute_initial_condition,
        name="ics")

    res = compute_initial_condition({'x':x})
    
    # Plot initial conditions 
    # fig = plt.figure(figsize=(20,10), dpi=150,num=1,clear=True)
    # fig.suptitle('ML Predicted Quantities', fontsize=16)
    # ax1 = fig.add_subplot(311) # Plot of u
    # ax1.plot(x, res['p'])
    # ax1.set_xlabel('x direction')
    # ax1.set_ylabel('Normalized Pressure')


    # ax2 = fig.add_subplot(312) # Plot of v
    # ax2.plot(x,res['u'])
    # ax2.set_xlabel('x direction')
    # ax2.set_ylabel('Normalized Velocity')
    
    # ax3 = fig.add_subplot(313) # Plot of v
    # ax3.plot(x,res['rho'])
    # ax3.set_xlabel('x direction')
    # ax3.set_ylabel('Normalized Density')
    # plt.show()

    pde = shocktube_pde(inputs=('t', 'x'), outputs=('p','u','rho'),gamma=config['gamma'],CFL=config['CFL'])

    """
        Set the wall boundary condition. Wave doesn't actually hit the wall. Solution is stopped before this happens
    """
    # left = Dirichlet(
    #     RandomSampler({'t': [ 0, config['tmax'] ], 'x': 0}, device=device, n_samples=1000), 
    #     lambda inputs: {
    #                     'p': torch.as_tensor(settings["u"]["min"]+0*inputs['x']).to(device),
    #                     'u': torch.as_tensor(settings["v"]["min"]+0*inputs['x']).to(device),
    #                     'rho': torch.as_tensor(settings["v"]["min"]+0*inputs['x']).to(device)
    #                 }, name="left-boundary"
    # ) # Left

    # right = Dirichlet(
    #     RandomSampler({'t': [ 0, config['tmax'] ], 'x': settings['xmax']}, device=device, n_samples=1000), 
    #     lambda inputs: {
    #                     'p': torch.as_tensor().to(device),
    #                     'u': torch.as_tensor().to(device),
    #                     'rho': torch.as_tensor().to(device)
    #                 }, name="right-boundary"
    # ) # Right

    """
        Solving the PDE
    """
    pde_sampler = RandomSampler({
        't': [0, config['tmax']],
        'x': [0, settings['xmax']],
    },device=device, n_samples=1000)

    pde.set_sampler(pde_sampler)
    pde.add_boco(initial_conditions)
    
    # solve
    LR = 1e-5
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

    config_name = config['name']
    # save the model
    torch.save({'model':mlp.state_dict(),
                'optimizer':optimizer.state_dict(),
                'history':hist,
                'num_inputs':n_inputs,
                'num_outputs':n_outputs,
                'n_layers':n_layers,
                'neurons':neurons,
                'settings':settings,
                }, f'{config_name}.pt')

    
    # 'scalers': {'x_scaler':x_scaler,'y_scaler':y_scaler,'u_scaler':u_scaler,'v_scaler':v_scaler}
    
    # plot_domain_2D('burgers_2D',x,y,u_history[-1],v_history[-1])
    # plot_history_2D('burgers_2D.gif',u_history,v_history,x,y)