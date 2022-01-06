from tensorflow.python.ops.gen_math_ops import Mul
import torch 
from torch import autograd
from MultiLayerLinear import MultiLayerLinear
import numpy as np 

device = "cuda" if torch.cuda.is_available() else "cpu"



def burgers_ml_solver(x:np.array,y:np.array,u:np.array,v:np.array,boundary_indicies:np.ndarray):
    """Solves the burgers equation from an array of points

    Args:
        x (np.array): 1D array containing all the values of x for all the points. Size=(N,)
        y (np.array): 1D array containing all the values of y for all the points. Size=(N,)
        u (np.array): 1D array containing all the values of u for all the points. Size=(N,)
        v (np.array): 1D array containing all the values of v for all the points. Size=(N,)
        boundary_indicies (np.ndarray): numpy array of all the indices, example [0,32,1,56,...,etc] where the boundaries are
    """
    pass 

def compute_pde_loss(vars,grads):
    pass    

def burgers_pde(inputs:torch.Tensor, outputs:torch.Tensor):
    """[summary]

    Args:
        inputs (torch.Tensor): column vectors of [x,y,t]
        outputs (torch.Tensor): u,v
    """
    

    grads, = autograd.grad(outputs, inputs, grad_outputs=outputs.data.new(outputs.shape).fill_(1),create_graph=True, only_inputs=True)
                                            # Computes the gradients consisting of du, dv, dx, dy
                                            # you can get du/dx simply by dividing the gradients 
    grads[:,]

    pde1 = nu * (d2u_dx2 + d2u_dy2) - (du_dt + u * du_dx + v * du_dy)
    pde1 = nu * (d2v_dx2 + d2v_dy2) - (dv_dt + u * dv_dx + v * dv_dy)


'''
    To represent burgers equation you have the following 
        inputs: u, v, x,y, xdot, ydot
        outputs: du,dv
'''
burgers_model = MultiLayerLinear(in_channels=3,out_channels=2,h_sizes=[128,128,128,128])