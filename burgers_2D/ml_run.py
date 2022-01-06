import sys
sys.path.insert(0,'../../')
from burgers_lib import initialize, create_domain, plot_domain_2D, plot_history_2D
from ml.burgers_ml_solver import burgers_pde
from nangs import Dirichlet
import numpy as np 
import json 

with open('settings.json','r') as f:
    settings = json.load(f)
    settings = settings['Burgers2D']
    x,y,u,v = create_domain(nx=settings['nx'],ny=settings['ny'])
    t = np.array(list(range(0,settings['tmax'],step=settings['dt'])))
    u_bounds = settings['u_bounds']
    v_bounds = settings['v_bounds']
    u,v = initialize(u,v,settings['u_max'],settings['v_max'],
                [u_bounds['i_percent'],u_bounds['j_percent']],
                [v_bounds['i_percent'],v_bounds['j_percent']])
    
    wall_boundaries = np.ma.make_mask(np.zeros(x.shape))
    # i advances in the x direction here we set bottom and top boundaries to True which means it's a wall
    wall_boundaries[:,0] = True # [i, j]    
    wall_boundaries[:,-1] = True
    
    # j advances in the y direction here we set left and right boundaries to True which means it's a wall
    wall_boundaries[0,:] = True # [i, j]
    wall_boundaries[-1,:] = True

    """
        Initialization:
        - The inputs are the denominators of the physics equation. 
        - Outputs are the quantities of interest. For the Burger's equation we want 
    """
    initial_conditions = {  
                            'inputs': {
                                        'x':x, 'y':y, 't':t
                                    },
                            'outputs': {
                                        'u':u, 'v':v
                                    }
                        }


    pde = burgers_pde(inputs=('x', 'y', 't'), outputs=('u', 'v'))

    pde.add_boco(initial_conditions)

    initial_condition = Dirichlet(
        RandomSampler({'x': [0., 1.], 'y': [0., 1.], 't': 0.}, device=device, n_samples=n_samples), 
        lambda inputs: {'p' : torch.sin(2.*np.pi*inputs['x'])*torch.sin(2.*np.pi*inputs['y'])},
        name="initial"
    )


    # plot_domain_2D('burgers_2D',x,y,u_history[-1],v_history[-1])
    plot_history_2D('burgers_2D.gif',u_history,v_history,x,y)