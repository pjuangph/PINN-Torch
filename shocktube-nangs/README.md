# Shock Tube
The Shock Tubes are devices that used to evaluate the responsiveness of pressure transducers. The design of a shock tube is simple. There's a long pipe with a diaphragm separating high pressure gas and low pressure gas. This diaphragm can be a rupture disk that burst at a certain pressure; when this happens there is rapid expansion of gasses from high pressure to low pressure creating a traveling shock wave. 

![Shock tube](shock_tube_diaphragm.png)

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


## Lax-Wendroff Time marching
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


## Roe Upwind Scheme 
Roe and upwind schemes in general must rely on characteristic theory to determine the direction of propagation of information to discretize the spatial derivatives. 

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



