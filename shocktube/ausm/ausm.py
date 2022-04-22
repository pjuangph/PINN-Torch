import numpy as np 

def flux_ausm(q:np.ndarray,gamma:float):
    """AUSM Flux Splitting Scheme
    
    References:
        Liou, M. S., & Steffen Jr, C. J. (1993). A new flux splitting scheme. Journal of Computational physics, 107(1), 23-39.
        https://github.com/PauloBarrosCorreia/AUSM/blob/master/ausm.f90 

    Args:
        q (np.ndarray): [3,nx] [rho, rhou, rhoE]
        gamma (float): ratio of specific heats 

    Returns:
        np.ndarray: Flux
    """
    # AUSM Mach Number 
    r = q[0,:]
    u = q[1,:]/r
    E = q[2,:]/r  # E = T*Cv # Kenji says maybe use Temperature instead of Energy then convert to Energy 

    P=(gamma-1.)*r*(E-0.5*u**2)
    a = np.sqrt(gamma*P/r)      
    M = u/a                     # Computes mach number at each location on the grid
    
    def M_plus_minus(M:float):
        """Equation 5

        Args:
            M (float): _description_

        Returns:
            _type_: _description_
        """
        if np.abs(M) <= 1:
            M_plus = 0.25*(M+1)
            M_minus = -0.25*(M-1)
        else:
            M_plus = 0.5*(M+np.abs(M))
            M_minus = 0.5*(M-np.abs(M))
        return M_plus, M_minus
    
    def P_plus_minus3(P:float, M:float):
        """Equation 8a

        Args:
            P (float): Pressure
            M (float): Mach 
        """
        if (np.abs(M)<=1):
            P_plus = 0.25*P * (M+1)**2 * (2-M)
            P_minus = 0.25*P * (M-1)**2 * (2+M)
        else:
            P_plus = 0.5*P * (M+np.abs(M))/M 
            P_minus = 0.5*P * (M-np.abs(M))/M 
        return P_plus, P_minus

    def P_plus_minus1(P:float,M:float):
        """Equation 8b

        Args:
            P (float): Pressure
            M (float): Mach
        """
        if (np.abs(M)<=1):
            P_plus = 0.5*P * (1+M)
            P_minus = 0.5*P * (1-M) 
        else:
            P_plus = 0.5*P * (M+np.abs(M))/M
            P_minus = 0.5*P * (M-np.abs(M))/M
        return P_plus, P_minus
    
    H = E+P/r
    F_half = np.zeros((3,q.shape[1]-2)) # rho*u, rho*u*u+P, rho*u*H
    F_L = np.zeros((3,))
    F_R = np.zeros((3,))
    for i in range(1,q.shape[1]-1):
        ML_plus,_ = M_plus_minus(M[i-1])
        _,MR_minus = M_plus_minus(M[i+1])
        M_half = ML_plus + MR_minus 
        
        PL_plus,_ = P_plus_minus1(P[i-1],M[i-1])
        _,PR_minus = P_plus_minus1(P[i+1],M[i+1])
        
        F_L[0] = r[i-1] * a[i-1]
        F_L[1] = r[i-1]*a[i-1]*u[i-1] 
        F_L[2] = r[i-1]*a[i-1]*H[i-1]

        F_R[0] = r[i+1] * a[i+1]
        F_R[1] = r[i+1]*a[i+1]*u[i+1] 
        F_R[2] = r[i+1]*a[i+1]*H[i+1]
        delta = F_R - F_L
        F_half[:,i-1] = M_half * 0.5*(F_L + F_R) - 0.5*np.abs(M_half)*delta+ (PL_plus - PR_minus)
    
    return F_half

def flux_ausm_np(q:np.ndarray,gamma:float):
    """AUSM Flux Splitting Scheme
    
    References:
        Liou, M. S., & Steffen Jr, C. J. (1993). A new flux splitting scheme. Journal of Computational physics, 107(1), 23-39.
        https://github.com/PauloBarrosCorreia/AUSM/blob/master/ausm.f90 

    Args:
        q (np.ndarray): [3,nx] [rho, rhou, rhoE]
        gamma (float): ratio of specific heats 

    Returns:
        np.ndarray: Flux
    """
    # AUSM Mach Number 
    r = q[0,:]
    u = q[1,:]/r
    E = q[2,:]/r  # E = T*Cv # Kenji says maybe use Temperature instead of Energy then convert to Energy 

    P=(gamma-1.)*r*(E-0.5*u**2)
    a = np.sqrt(gamma*P/r)      
    M = u/a                     # Computes mach number at each location on the grid 

    M_plus = np.less_equal(np.abs(M),1)*0.25*(M+1)**2 + \
                np.greater(np.abs(M),1) * 0.5*(M+abs(M)) # M+, np.less_than/np.greater produces a boolean array [0,0,0,1,1,1] etc 

    M_neg = np.less_equal(np.abs(M),1)*(-0.25*(M-1)**2) + \
                np.greater(np.abs(M),1)*0.5*(M-abs(M)) # M- 
    
    M_half = M_plus[1:] + M_neg[0:-1]  # (M+ index 1 to end) + (M- index 0 to end-1)

    # AUSM Pressure

    P_plus = np.less_equal(M,1)*0.25*P*(M+1)*(M+1) * (2.0 - M) \
                + np.greater(M,1)*np.nan_to_num((0.5*P*(M+np.abs(M)) / M),nan=0)

    P_neg = np.less_equal(M,1)*0.25*P*(M-1)*(M-1) * (2.0 + M) \
                + np.greater(M,1)*np.nan_to_num(0.5*P*(M-np.abs(M)) / M,nan=0)
        
    # P_half = P_plus[1,:] + P_neg[0:-1]
    
    H = E+P/r
    F_L = np.stack([r*a, r*a*u, r*a*H])[:,0:-1]
    F_R = np.stack([r*a, r*a*u, r*a*H])[:,1:]
    
    # AUSM Discretization
    F_half = M_half * 0.5 * ( F_L + F_R ) - 0.5 * np.abs(M_half) * (F_R-F_L) + P_plus[0:-1] + P_neg[1:] # Equation 9 from AUSM Paper 

    return F_half # F
