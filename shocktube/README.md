# Shock Tube
The Shock Tubes are devices that used to evaluate the responsiveness of pressure transducers. The design of a shock tube is simple. There's a long pipe with a diaphragm separating high pressure gas and low pressure gas. This diaphragm can be a rupture disk that burst at a certain pressure; when this happens there is rapid expansion of gasses from high pressure to low pressure creating a traveling shock wave. 

![Shock tube](https://static-01.hindawi.com/articles/ijae/volume-2017/2010476/figures/2010476.fig.001.svgz)

# Governing Equations

```math
\frac{\partial}{\partial t} \begin{bmatrix} 
\rho \\
\rho u \\ 
rho E
\end{bmatrix} 
+
\frac{\partial}{\partial x} \begin{bmatrix}
\rho u \\
\rho u^2 + P \\
(rhoE + P) * u
\end{bmatrix} = 0
```


```math
F(U) = \begin{bmatrix}
\rho u \\
\rho u^2 + P \\
(rhoE + P) * u
\end{bmatrix}
```

Initial speed of sound
```math
a_0 = \sqrt(\gamma * P_0 / \rho )
```

$u_0$ is a vector defining the initial velocity in the system
```math
dt = CFL * dx / max(|u_0| + a_0)
```

Where 
```math
P = \rho*(\gamma - 1)(E - 0.5u^2)
```


and $\gamma = 1.4$ 

Internal Energy is 
```math
e = c_v * T 
```

```math
c_v = \frac{k}{\mu} * \frac{1}{\gamma-1}
```

Viscosity $\mu$ and thermal conductivity $k$ for air needs to be looked up. Units of $c_v$ is J/(Kg K)

## Analytical Solutions
### Lax-Wendroff Time marching
Predictor step
```math
\tilde{U}_{j+0.5} = \frac{U_j^n + U_{j+1}^n}{2} - \frac{\Delta t }{2\Delta x} \big[F(U)_{j+1}^n - F(U)_{j}^n \big]
```

Corrector
```math
u_{j}^{n+1} = U_j - \frac{\Delta t}{\Delta x} \big[ F(\tilde{U_{j+0.5}}^n - F(\tilde{U_{j-0.5}}^n \big]
```

Note:
> Solution must be stopped before the wave hits the boundary


### Roe Upwind Scheme 
Roe and upwind schemes in general must rely on characteristic theory to determine the direction of propagation of information to discretize the spatial derivatives. 

### AUSM Scheme
To see the implementation of AUSM go to [link](https://gitlab.grc.nasa.gov/ideas/pinn-torch/-/tree/main/shocktube/ausm)

## PINNs based solutions
1. Pedro, J. B., Maroñas, J., & Paredes, R. (2019). Solving partial differential equations with neural networks. arXiv preprint arXiv:1912.04737. https://github.com/juansensio/nangs 
    - Paht's implementation and comments https://github.com/pjuangph/nangs
    
2. Michoski, C., Milosavljević, M., Oliver, T., & Hatch, D. R. (2020). Solving differential equations using deep neural networks. Neurocomputing, 399, 193-212. https://github.com/Gillgren/Solving-Differential-Equations-with-Neural-Networks 

3. Patel, R. G., Manickam, I., Trask, N. A., Wood, M. A., Lee, M., Tomas, I., & Cyr, E. C. (2022). Thermodynamically consistent physics-informed neural networks for hyperbolic systems. Journal of Computational Physics, 449, 110754. https://github.com/rgp62/cvpinns 
    - Paht's implementation and comments: https://github.com/pjuangph/cvpinns  

4. Papados, A. Solving Hydrodynamic Shock-Tube Problems Using Weighted Physics-Informed Neural Networks with Domain Extension. https://github.com/alexpapados/Physics-Informed-Deep-Learning-Solid-and-Fluid-Mechanics 
    - Paht's implementation and comments https://github.com/pjuangph/Physics-Informed-Deep-Learning-Solid-and-Fluid-Mechanics
 




# Application of nangs to Shocktube

The application of nangs doesn't quite work. Here's why I think it fails. The MLP (Multilayer perception) used in nangs is very basic. They assumed that this MLP is good for solving multiple problems, it is. 
nangs PDE https://github.com/juansensio/nangs/blob/master/nangs/nn/mlp.py 

```python
class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def block(i, o):
    fc = torch.nn.Linear(i, o)
    return torch.nn.Sequential(
        Sine(),
        torch.nn.Linear(i, o)
    )


class MLP(torch.nn.Module):
    def __init__(self, inputs, outputs, layers, neurons):
        super().__init__()
        fc_in = torch.nn.Linear(inputs, neurons)
        fc_hidden = [
            block(neurons, neurons)
            for layer in range(layers-1)
        ]
        fc_out = block(neurons, outputs)

        self.mlp = torch.nn.Sequential(
            fc_in,
            *fc_hidden,
            fc_out
        )

    def forward(self, x):
        return self.mlp(x)
```

changing `sin` to `cos` significantly alters the result. I was not able to reproduce the initial conditions at t=0. If this fails then the rest of it fails as well.



