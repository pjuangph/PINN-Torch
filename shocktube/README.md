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
\rho v \\
\rho u^2 + P \\
(rhoE + P) * u
\end{bmatrix} = 0
```


```math
F(U) = \begin{bmatrix}
\rho v \\
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



