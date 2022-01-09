'''
    Code below runs the analytical solution for burgers 2D 
'''

from burgers_lib import initialize_uv, create_domain, plot_domain_2D, plot_history_2D
from analytical.burgers_solver_2D import burgers_timestep
import json 
from tqdm import trange

with open('settings.json','r') as f:
    settings = json.load(f)
    settings = settings['Burgers2D']
    x,y,u,v = create_domain(nx=settings['nx'],ny=settings['ny'],xmax=settings['x']['max'],ymax=settings['y']['max'])
    u_bounds = settings['u_bounds']
    v_bounds = settings['v_bounds']
    u,v = initialize_uv(x,y,u,v,settings['u']['max'],settings['v']['max'],
                settings['u']['min'],settings['v']['min'],
                u_bounds,v_bounds)
    
    u_history, v_history = burgers_timestep(x,y,u,v,nt=int(settings['tmax']/settings['dt']),dt=settings['dt'],nu=settings['nu'])
    print("plotting figures")
    # Plot results at a certain point in time
    for i in trange(len(u_history)):
        plot_domain_2D('analytical',x,y,u_history[i],v_history[i],i)
    