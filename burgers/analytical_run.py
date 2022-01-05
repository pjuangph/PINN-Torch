'''
    Code below runs the analytical solution for burgers 2D 
'''

from burgers_lib import initialize, create_domain, plot_domain_2D, plot_history_2D
from analytical.burgers_solver_2D import burgers_timestep

x,y,u,v = create_domain(nx=80,ny=80)
u,v = initialize(u,v,5,5,[(0.2,0.8),(0.1,0.9)],[(0.2,0.8),(0.1,0.9)])
u_history, v_history = burgers_timestep(x,y,u,v,50,0.01,nu=0.1)

plot_history_2D('burgers_2D.gif',u_history,v_history,x,y)