'''
    Code below runs the analytical solution for burgers 2D 
'''

from burgers_lib import initialize, create_domain, plot_domain_2D, plot_history_2D
from analytical.burgers_solver_2D import burgers_timestep
import json 

with open('settings.json','r') as f:
    settings = json.load(f)
    settings = settings['Burgers2D']
    x,y,u,v = create_domain(nx=settings['nx'],ny=settings['ny'])
    u_bounds = settings['u_bounds']
    v_bounds = settings['v_bounds']
    u,v = initialize(u,v,settings['u']['max'],settings['v']['max'],
                [u_bounds['i_percent'],u_bounds['j_percent']],
                [v_bounds['i_percent'],v_bounds['j_percent']])
    u_history, v_history = burgers_timestep(x,y,u,v,nt=500,dt=0.001,nu=0.1)
    # Plot results at a certain point in time
    plot_domain_2D('burgers_2D',x,y,u_history[-1],v_history[-1])
    plot_history_2D('burgers_2D.gif',u_history,v_history,x,y)