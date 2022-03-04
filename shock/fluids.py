from typing import Dict
import numpy as np 
def sutherland(T:np.ndarray) -> np.ndarray:
    """Sutherland law for computing viscosity of air 

    Args:
        T (float): Temperature in Kelvin

    Returns:
        dynamic_viscosity: Units Pa*s
    """
    mu0 =  1.716E-5
    T0 = 273
    Smu = 111
    return mu0 * (T/T0)**1.5 * (T0+Smu)/(T+Smu)

def internal_energy(T:float,gamma:float=1.4):
    """Calculates the internal energy

    Args:
        T (float): Temperature in Kelvin
        gamma (float, optional): Ratio of specific heats Cp/Cv. Defaults to 1.4.

    Returns:
        float: Internal energy in J/(kg K)
    """
    mu = sutherland(T)

    k=1.38E-23 # J/K boltzman constant
    return k*T/(mu*(gamma-1))

def pressure(rho:float,T:float,R:float=287.05):
    return rho*R*T