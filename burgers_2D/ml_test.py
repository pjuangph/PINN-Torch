'''
    ml_test.py
    This file reads the simulation model and predicts the value of u and v at random x and y points for a given time and compares it to the analytical solution
'''
import torch, sys
sys.path.insert(0,'../../nangs')
from nangs import MLP 
import os.path as osp 
import json 
from burgers_lib import create_domain, plot_domain_2D, plot_history_2D
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    x = torch.tensor(x,dtype=torch.float32).to(device)
    y = torch.tensor(y,dtype=torch.float32).to(device)
    t = torch.tensor(t,dtype=torch.float32).to(device)
    input = torch.stack((x,y,t),dim=1)
    out = model(input)
    out=out.cpu().numpy()
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

        x_original,y_original,u,v = create_domain(nx=settings['nx'],ny=settings['ny'],xmax=settings['x']['max'],ymax=settings['y']['max'])
        X,Y = np.meshgrid(x_original,y_original)

    t = np.arange(0,settings['tmax'],settings['dt']) # user will change this 

    if osp.exists('burgers_train.pt'):
        data = torch.load('burgers_train.pt')

    x = X.flatten()
    y = Y.flatten()
    model = MLP(data['num_inputs'],data['num_outputs'],data['n_layers'],data['neurons'])
    model.load_state_dict(data["model"])
    model.to(device)
    u_history = list(); v_history=list()
    for i in trange(len(t)):
        u,v = compute_results(model,x,y,t[i])
        u_history.append(u.reshape(X.shape[0],X.shape[1]))
        v_history.append(v.reshape(X.shape[0],X.shape[1]))
    print('creating figures')
    for i in trange(len(u_history)):
        plot_domain_2D('ml',X,Y,u_history[i],v_history[i],i)
    
