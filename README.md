## PINN-Torch Physics Informed Neural Network using PyTorch

The intent of this repository is to show how to solve example and test PDEs using neural networks. The goal is to provide excellent documentation for others to learn from and also create interactive tutorials that spin up on google colab. 

## Requirements

All requirements are in the `requirements.txt`. To get started, simply do `pip install -r requirements.txt` 



> ### Note: 
> **Displaying Math**: This repository contains a lot of math equations. Github currently does not display latex equations. If you want to see the equations, I reccomend installing xhub google chrome: [xhub](https://github.com/nschloe/xhub)

> **Creating GIFs**:To save gif files you need to have imagemagick installed and added to your **path** variable
https://imagemagick.org/script/download.php 

---

## List of Examples 

All of these examples are presented on a 2D Domain of x and y. The importance of the Physics based neural network is that it doesn't require a mesh in order to obtain a solution; all you need is enough samples within your domain.

- [Viscous Burgers](https://github.com/pjuangph/PINN-Torch/tree/main/burgers_2D): Burgers equation describes the viscous movement of fluid at each location on a 2D space as a function of time. 
- [Poisson Equation for Heat Conduction](https://github.com/pjuangph/PINN-Torch/tree/main/Poisson): This example shows how to use poisson's equation to describe the heat conduction through a plate 
- [Euler 2D](https://github.com/pjuangph/PINN-Torch/tree/main/euler): This example shows how neural networks to solve euler's equation in gas dynamics for a 2D flow field 

-[Shock Tube](https://github.com/pjuangph/PINN-Torch/tree/main/shocktube-nangs): Using the same PINNs framework `nangs` to solve shocktube. This doesn't work and shows the limiations of PINNs. NANGs is great a solving problems that are not continous. When you have something like this you may need a different library


# Useful references
1. [Solving PDES (nangs)](https://github.com/pjuangph/PINN-Torch/blob/main/references/Solving_PDE_with_NN.pdf) I used their code, nangs, in this repo quite often. I highly reccomend their work. 

2. [Shocktube not nangs](https://www.researchgate.net/profile/Alexandros-Papados/publication/350239546_Solving_Hydrodynamic_Shock-Tube_Problems_Using_Weighted_Physics-Informed_Neural_Networks_with_Domain_Extension/links/6130062f0360302a0073573c/Solving-Hydrodynamic-Shock-Tube-Problems-Using-Weighted-Physics-Informed-Neural-Networks-with-Domain-Extension.pdf)  Link to the code [Alex Papados](https://github.com/alexpapados/Physics-Informed-Deep-Learning-Solid-and-Fluid-Mechanics)

# Not so useful reference
1. [Physics based deep learning](https://github.com/pjuangph/PINN-Torch/blob/main/references/physics%20based%20deep%20learning.pdf) These guys have a website designed to make you click. They show different examples of way to solve a PDE but their website isn't as current as the research. 

