
import numpy as np 
import matplotlib.pyplot as plt 
import torch
from nangs import PDE
import math 

class burgers_pde(PDE):
    def computePDELoss(self,inputs,outputs):
        u, v = outputs[:, 0], outputs[:, 1]
        
        # Compute Derivatives
        grads = self.computeGrads(u,inputs)
        dudt = grads[:, 0]
        dudx = grads[:, 1]
        dudy = grads[:, 2]

        dvdt = grads[:, 3]
        dvdx = grads[:, 4]
        dvdy = grads[:, 5].

        dudt = grads[:, 0]
        dudx = grads[:, 1]
        dudy = grads[:, 2]

        dvdt = grads[:, 3]
        dvdx = grads[:, 4]
        dvdy = grads[:, 5]

# initialize         
pde = burgers_pde(inputs=('x', 'y', 't'), outputs=('u', 'v'))
