

# The Burgers Equation
Burger's equation is used to model the viscosity's effect on a flow through a pipe. The assumption is that there's no flow at the boundary. Boundary of the domain is assumed to be a fixed wall. The flow is only exists inside the domain. The effects of viscosity causes the velocity (u, v) to decay to 0 near the walls.  

For 1D: 
```math
\frac{du}{dt} + u\frac{du}{dx} = \nu \frac{d^2 u}{dx^2}
```
For -1.0 < x < +1.0, and 0.0 < t

For 2D:
```math
\frac{du}{dt} + u \frac{du}{dx} + v \frac{du}{dy}  = \nu ( \frac{d^2 u}{dx^2} + \frac{d^2 u}{dy^2} )
```

```math
\frac{dv}{dt} + u \frac{dv}{dx} + v \frac{dv}{dy}  = \nu ( \frac{d^2 v}{dx^2} + \frac{d^2 v}{dy^2} )
```

More info on 2D Burgers: http://cfdmoments.blogspot.com/2014/07/2d-burgers-equation.html 


# Burgers actual solution in 1D 
To solve the actual burger's equation call `python main_burgers_2D.py` This will initialize the burger's equation and solve and then plot the results

[Background on finite difference methods](https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf)

# Burgers Solution 2D
##Analytical Solution
Solving the analytical solution requires using finite differencing. Below is an example of **Euler time step** which is solving for future time step using previous one. Differencing in space is First order and central differencing is used for the second derivatives. 

See [burgers_timestep](https://github.com/pjuangph/PINN-Torch/blob/main/burgers_2D/analytical/burgers_solver_2D.py) 

```python
dx = x[0,2]-x[0,1] # We can do it this way because x and y are initialized using linspace which guarantees constant spacing 
dy = y[1,0]-y[0,0]

u_history = list()
v_history = list() 
u_history.append(copy.deepcopy(u)) # set equal to initial value
v_history.append(copy.deepcopy(v))
for n in trange(nt):
    un = u.copy()   # previous value
    vn = v.copy()
    for i in range(1,x.shape[0]-1):
        for j in range(1,y.shape[1]-1):
            # Uses backward difference in space to solve first order derivative
            # Central differencing for second order derivative 
            u[i,j] = (un[i, j] -(un[i, j] * dt / dx * (un[i, j] - un[i-1, j])) -vn[i, j] * dt / dy * (un[i, j] - un[i, j-1])) + (nu*dt/(dx**2))*(un[i+1,j]-2*un[i,j]+un[i-1,j])+(nu*dt/(dx**2))*(un[i,j-1]-2*un[i,j]+un[i,j+1])
            v[i,j] = (vn[i, j] -(un[i, j] * dt / dx * (vn[i, j] - vn[i-1, j]))-vn[i, j] * dt / dy * (vn[i, j] - vn[i, j-1])) + (nu*dt/(dx**2))*(vn[i+1,j]-2*vn[i,j]+vn[i-1,j])+(nu*dt/(dx**2))*(vn[i,j-1]-2*vn[i,j]+vn[i,j+1])

    u[:,0] = 1       # At all i values when j = 0
    u[:,-1] = 1      # At all i values when j = jmax
    u[0,:] = 1       # At all j values and i = 0
    u[-1,:] = 1     # At all j values and i = imax

    v[:,0] = 1
    v[:,-1] = 1
    v[0,:] = 1
    v[-1,:] = 1
```

![](https://github.com/pjuangph/PINN-Torch/blob/main/burgers_2D/analytical.gif)

##ML Solution
Solving the machine learning solution is a bit difference different. There's no finite differencing, instead you define your loss equation of your PDE



```python
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
```
See: 
- [Training code](https://github.com/pjuangph/PINN-Torch/blob/main/burgers_2D/ml_train.py)
- [Burgers PDE in ML](https://github.com/pjuangph/PINN-Torch/blob/main/burgers_2D/ml/burgers_pde.py)

![](https://github.com/pjuangph/PINN-Torch/blob/main/burgers_2D/ml.gif)

