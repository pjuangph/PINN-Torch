import sys
import numpy as np
import matplotlib.pyplot as plt 
import torch
sys.path.insert(0,'../../nangs')

from nangs import PDE
import math 

class burgers_pde(PDE):
    def __init__(self,inputs,outputs,nu:float=0.1):
        super().__init__(inputs,outputs)
        """Initializes Burger's PDE 

        Args:
            nu (float, optional): [description]. Defaults to 0.1.
        """
        self.nu = nu 

    def computePDELoss(self,inputs:torch.Tensor,outputs:torch.Tensor):
        """Compute the loss in burger's equation 

        Args:
            inputs (torch.Tensor): x, y, t as tensors with shape of (npoints, 3)
            outputs (torch.Tensor): this is u and v as tensors with shape of (npoints,2)
        """        
        # To compute du_dx, du_dy we have to extract u and v from the outputs and use them in the gradients
        u, v = outputs[:,0], outputs[:,1]
        
        # We want the output to be u and the input to be x,y,z this computes the gradient for du_dx, du_dy, du_dt
        grads = self.computeGrads(u, inputs) # output, input
        du_dx = grads[:,0]; du_dy = grads[:,1]; du_dt = grads[:,2]

        grads = self.computeGrads(v, inputs) # output, input
        dv_dx = grads[:,0]; dv_dy = grads[:,1]; dv_dt = grads[:,2]

        # Compute the gradients of u
        ddu_dxx = self.computeGrads(du_dx,inputs)[:,0] # du_dx is output, x is input -> computes ddu_dxx 
        ddu_dyy = self.computeGrads(du_dy,inputs)[:,1]
        
        # Compute the gradients of v
        ddv_dxx = self.computeGrads(dv_dx,inputs)[:,0]
        ddv_dyy = self.computeGrads(dv_dy,inputs)[:,1]

        # Burgers PDE
        return { 
            'u-pde': du_dt + u*du_dx + v*du_dy - self.nu * (ddu_dxx+ ddu_dyy),
            'v-pde': dv_dt + u*dv_dx + v*dv_dy - self.nu * (ddv_dxx+ ddv_dyy) }