'''
    ml_test.py
    This file reads the simulation model and predicts the value of u and v at random x and y points for a given time and compares it to the analytical solution
'''
import torch, sys
sys.path.insert(0,'../../nangs')
from nangs import Dirichlet, MLP 
import os.path as osp 
import json 
from burgers_lib import initialize, create_domain, plot_domain_2D, plot_history_2D
import numpy as np 
from sklearn.preprocessing import MinMaxScaler

@torch.no_grad()
def compute_results(model:torch.nn.Module,x:np.ndarray,y:np.ndarray,t:float):
    """[summary]

    Args:
        model (torch.nn.Module): model 
        X (np.ndarray): vector containing x coordinates
        Y (np.ndarray): vector containing y coordinates 
        t (float): time in seconds

    Returns:
        (tuple): containing

            **u** (np.ndarray): u velocity
            **v** (np.ndarray): v velocity
    """
    # Create the inputs 
    
    t = x*0+t 
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    t = torch.tensor(t,dtype=torch.float32)
    input = torch.stack((x,y,t),dim=1)
    out = model(input).detach().numpy()
    return out[:,0], out[:,1]
    

if __name__=="__main__":
    

    assert osp.exists('burgers_train.pt'), "You need to train the model before you can plot the results"
    assert osp.exists('settings.json'), "Need the settings file that defines the setup conditions"

    ''' 
        Load the settings files
    '''
    with open('settings.json','r') as f:
        settings = json.load(f)
        settings = settings['Burgers2D']

        x_original,y_original,u,v = create_domain(nx=settings['nx'],ny=settings['ny'])
        X,Y = np.meshgrid(x_original,y_original)

    t = np.arange(0,settings['tmax'],100) # user will change this 

    if osp.exists('burgers_train.pt'):
        data = torch.load('burgers_train.pt')
    #     x_scaler = data['scalers']['x_scaler']
    #     y_scaler = data['scalers']['y_scaler']
    #     u_scaler = data['scalers']['u_scaler']
    #     v_scaler = data['scalers']['v_scaler']
    # # Normalize
    # x = x_scaler.fit_transform(X.flatten().reshape(-1,1))[:,0]
    # y = y_scaler.fit_transform(Y.flatten().reshape(-1,1))[:,0]

    x = X.flatten()
    y = Y.flatten()
    model = MLP(data['num_inputs'],data['num_outputs'],5,128)
    for i in range(len(t)):
        u,v = compute_results(model,x,y,t[i])

        u = u.reshape(X.shape[0],X.shape[1])
        v = v.reshape(X.shape[0],X.shape[1])
        plot_domain_2D('ml',x_original,y_original,u,v)
    
