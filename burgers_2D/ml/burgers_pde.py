import sys
import numpy as np
import matplotlib.pyplot as plt 
import torch
from nangs import PDE
import math 

class burgers_pde(PDE):
    def computePDELoss(self,inputs:torch.Tensor,outputs:torch.Tensor, **kwargs):
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
        ddu_dxx = self.computeGrads(du_dx,inputs[:,0]) # du_dx is output, x is input -> computes ddu_dxx 

        ddu_dyy = self.computeGrads(du_dy,inputs[:,1])
        
        # Compute the gradients of v
        ddv_dxx = self.computeGrads(dv_dx,inputs[:,0])
        ddv_dyy = self.computeGrads(dv_dy,inputs[:,1])

        # Burgers PDE
        return { 
            'u_velocity': du_dt + u*du_dx + v*du_dy - kwargs['mu'] * (ddu_dxx+ ddu_dyy),
            'v_velocity': dv_dt + u*dv_dx + v*dv_dy - kwargs['mu'] * (ddv_dxx+ ddv_dyy) }