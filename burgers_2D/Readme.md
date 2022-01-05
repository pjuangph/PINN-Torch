

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

# Burgers actual solution 2D


# Burgers - Solution using Physics based neural networks

