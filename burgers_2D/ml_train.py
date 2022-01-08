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
    settings['tmax'] = 1
    t = np.arange(0,1,0.02) # settings['tmax'] = 0.001 is too agressive
    u_bounds = settings['u_bounds']
    v_bounds = settings['v_bounds']
    
    # u_init,v_init = initialize(u,v,settings['u']['max'],settings['v']['max'],
    #             settings['u']['min'],settings['v']['min'],
    #             [u_bounds['i_percent'],u_bounds['j_percent']],
    #             [v_bounds['i_percent'],v_bounds['j_percent']])
    
    # Vectorize, Normalize, convert back
    # x_scaler = MinMaxScaler()
    # y_scaler = MinMaxScaler()
    # X = np.reshape(x_scaler.fit_transform(X.flatten().reshape(-1,1)),(nx,ny))
    # Y = np.reshape(y_scaler.fit_transform(Y.flatten().reshape(-1,1)),(nx,ny))

    # u_scaler = MinMaxScaler()
    # v_scaler = MinMaxScaler()
    # u_init = np.reshape(u_scaler.fit_transform(u_init.flatten().reshape(-1,1)),(nx,ny))
    # v_init = np.reshape(v_scaler.fit_transform(v_init.flatten().reshape(-1,1)),(nx,ny))

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

        return {'u': u.type(torch.FloatTensor),
                'v':v.type(torch.FloatTensor)}


    initial_conditions = Dirichlet(
        RandomSampler({'x': [0., 2.], 'y': [0., 2.], 't': 0}, device=device, n_samples=1000), 
        lambda inputs: initial_conditions_func(inputs),
        name="initial-conditions"
    )

    pde = burgers_pde(inputs=('x', 'y', 't'), outputs=('u', 'v'),nu=settings['nu'])

    """
        Set the wall boundary condition
    """
    wall_bottom = Dirichlet(
        RandomSampler({'x': [0., 2.], 'y': 0., 't': [0, settings['tmax']]}, device=device, n_samples=1000), 
        lambda inputs: {'u': torch.as_tensor(0*inputs['x']), 'v': torch.as_tensor(0*inputs['x'])},
        name="wall-bottom"
    )

    wall_top = Dirichlet(
        RandomSampler({'x': [0., 2.], 'y': 2., 't': [0, settings['tmax']]}, device=device, n_samples=1000), 
        lambda inputs: {'u': torch.as_tensor(0*inputs['x']), 'v': torch.as_tensor(0*inputs['x'])},
        name="wall-top"
    )

    wall_left = Dirichlet(
        RandomSampler({'x': 0., 'y': [0, 2.], 't': [0, settings['tmax']]}, device=device, n_samples=1000), 
        lambda inputs: {'u': torch.as_tensor(0*inputs['x']), 'v': torch.as_tensor(0*inputs['x'])},
        name="wall-left"
    )

    wall_right = Dirichlet(
        RandomSampler({'x': 2., 'y': [0, 2.], 't': [0, settings['tmax']]}, device=device, n_samples=1000), 
        lambda inputs: {'u': torch.as_tensor(0*inputs['x']), 'v': torch.as_tensor(0*inputs['x'])},
        name="wall-right"
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
    u_wall = torch.as_tensor(t_wall*0,dtype=torch.float32,device=device) # velocity in u is set to 0
    v_wall = torch.as_tensor(t_wall*0,dtype=torch.float32,device=device)

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
    
    X_sample = np.reshape(X,(X.shape[0],X.shape[1],1))
    Y_sample = np.reshape(Y,(Y.shape[0],Y.shape[1],1))
    X_sample = np.repeat(X_sample,len(t),axis=2)
    Y_sample = np.repeat(Y_sample,len(t),axis=2)
    t_sample = X_sample.copy()
    for i in range(len(t)):
        t_sample[:,:,i] = t[i]

    # pde_sampler = Custom_Sampler({
    #     'x': torch.as_tensor(X_sample.flatten(),device=device,dtype=torch.float32), 
    #     'y': torch.as_tensor(Y_sample.flatten(),device=device,dtype=torch.float32),
    #     't': torch.as_tensor(t_sample.flatten(),device=device,dtype=torch.float32)
    # }, device=device, n_samples=len(t_sample.flatten()))           # Here we are solving the entire equation for all times 1000 samples

    pde_sampler = RandomSampler({
        'x': [0.,2.],
        'y': [0.,2.],
        't': [0.,1.]
    },device=device, n_samples=5000)

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
    mlp = MLP(n_inputs,n_outputs,5,100).to(device) # MultiLayerLinear(n_inputs, n_outputs, hidden_layers).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters())
    scheduler = None # torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.5, pct_start=0.3, div_factor=5, final_div_factor=2, total_steps=n_steps)

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