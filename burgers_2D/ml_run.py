
from burgers_lib import initialize, create_domain, plot_domain_2D, plot_history_2D
from ml.burgers_ml_solver import 
import json 

with open('settings.json','r') as f:
    settings = json.load(f)
    settings = settings['Burgers2D']
    x,y,u,v = create_domain(nx=settings['nx'],ny=settings['ny'])
    u_bounds = settings['u_bounds']
    v_bounds = settings['v_bounds']
    u,v = initialize(u,v,settings['u_max'],settings['v_max'],
                [u_bounds['i_percent'],u_bounds['j_percent']],
                [v_bounds['i_percent'],v_bounds['j_percent']])
    



    # plot_domain_2D('burgers_2D',x,y,u_history[-1],v_history[-1])
    plot_history_2D('burgers_2D.gif',u_history,v_history,x,y)