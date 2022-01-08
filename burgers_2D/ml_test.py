'''
    ml_test.py
    This file reads the simulation model and predicts the value of u and v at random x and y points for a given time and compares it to the analytical solution
'''
import torch
from ml.MultiLayerLinear import MultiLayerLinear
import os.path as osp 
import json 
from burgers_lib import initialize, create_domain, plot_domain_2D, plot_history_2D
import numpy as np 

@torch.no_grad()
def compute_results(model:torch.nn.Module,t:float,X:np.ndarray,Y:np.ndarray):
    
if __name__=="__main__":
    

    assert osp.exists('burgers_train.pt'), "You need to train the model before you can plot the results"
    assert osp.exists('settings.json'), "Need the settings file that defines the setup conditions"

    ''' 
        Load the settings files
    '''
    with open('settings.json','r') as f:
        settings = json.load(f)
        settings = settings['Burgers2D']
        x,y,u,v = create_domain(nx=settings['nx'],ny=settings['ny'])
        X,Y = np.meshgrid(x,y)
        
    t = range(0,settings['tmax'],100) # user will change this 
    time_tensor = X*0 + t


    if osp.exists('burgers_train.pt'):
        data = torch.load('burgers_train.pt')
    
    model = MultiLayerLinear(data['num_inputs'],data['num_outputs'],data['hidden_layers'])
    
    model.eval()
    
