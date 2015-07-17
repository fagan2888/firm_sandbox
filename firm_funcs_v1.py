'''
------------------------------------------------------------------------
Last updated 7/15/2015

This file contains functions for the producer's problem.
------------------------------------------------------------------------
'''
# Import Packages
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''



def firmsolve(K1L1vec, params, K, L):
    A, gamma, epsilon = params
    K1, L1 = K1L1vec
    y_params = np.array([A, gamma, epsilon])
    #Y1 = get_Y(y_params, K1, L1)
    #Y2 = get_Y(y_params, K-K1, L-L1)
    #focK = ((L1 ** (1 - gamma)) * (K1 ** (gamma - 1)) -
    #       ((L - L1) ** (1 - gamma)) * ((K - K1) ** (gamma - 1)))
    #focL = ((K1 ** gamma) * (L1 ** (-gamma)) -
    #       ((K - K1) ** gamma) * ((L - L1) ** (-gamma)))
    #focK = (((gamma*Y1)/K1)**(1/epsilon))-((((gamma)*Y2)/(K-K1))**(1/epsilon))
    #focL = ((((1-gamma)*Y1)/L1)**(1/epsilon))-((((1-gamma)*Y2)/(L-L1))**(1/epsilon))
    focK = (((A * (((gamma**(1/epsilon))*(K1**((epsilon-1)/epsilon))) + 
          (((1-gamma)**(1/epsilon))*(L1**((epsilon-1)/epsilon))))**(1/(epsilon-1)))*((gamma**(1/epsilon))*(K1**(-1/epsilon))))-
          ((A * (((gamma**(1/epsilon))*((K-K1)**((epsilon-1)/epsilon))) + 
          (((1-gamma)**(1/epsilon))*((L-L1)**((epsilon-1)/epsilon))))**(1/(epsilon-1)))*((gamma**(1/epsilon))*((K-K1)**(-1/epsilon)))))
    focL = (((A * (((gamma**(1/epsilon))*(K1**((epsilon-1)/epsilon))) + 
          (((1-gamma)**(1/epsilon))*(L1**((epsilon-1)/epsilon))))**(1/(epsilon-1)))*(((1-gamma)**(1/epsilon))*(L1**(-1/epsilon))))-
          ((A * (((gamma**(1/epsilon))*((K-K1)**((epsilon-1)/epsilon))) + 
          (((1-gamma)**(1/epsilon))*((L-L1)**((epsilon-1)/epsilon))))**(1/(epsilon-1)))*(((1-gamma)**(1/epsilon))*((L-L1)**(-1/epsilon)))))
    firmfocs = np.append(focK, focL)
    return firmfocs


def get_KL(params, L, K):
    A, gamma, epsilon, delta, SS_tol = params
    K1L1_guess = np.array([K / 2, L / 2])
    K1L1_params = np.array([A, gamma, epsilon])
    K1, L1 = opt.fsolve(firmsolve, K1L1_guess, args=(K1L1_params, K, L), xtol=SS_tol)
    return K1, L1



def get_Y(params, K, L):
    '''
    Generates aggregate output Y

    Inputs:
        params = [2,] vector, production function parameters [A, alpha]
        A      = scalar > 0, total factor productivity of production
                 function
        alpha  = scalar in (0,1), capital share of income in production
                 function
        K      = scalar > 0, aggregate capital stock
        L      = scalar > 0, aggregate labor

    Functions called: None

    Objects in function:
        Y = scalar > 0, aggregate output (GDP)

    Returns: Y
    '''
    A, gamma, epsilon = params
    #Y = A * (K ** gamma) * (L ** (1 - gamma))
    Y = (A * (((gamma**(1/epsilon))*(K**((epsilon-1)/epsilon))) + 
          (((1-gamma)**(1/epsilon))*(L**((epsilon-1)/epsilon))))**(epsilon/(epsilon-1)))
    return Y





def get_r(params, K, L):
    '''
    Generates real interest rate r from parameters, aggregate capital
    stock K, and aggregate labor L

    Inputs:
        params = [3,] vector, [A, alpha, delta]
        A      = scalar > 0, total factor productivity of production
                 function
        alpha  = scalar in (0,1), capital share of income in production
                 function
        delta  = scalar in [0,1], model-period depreciation rate of
                 capital
        K      = scalar > 0, aggregate capital stock
        L      = scalar > 0, aggregate labor

    Functions called: None

    Objects in function:
        r = scalar > 0, real interest rate (return on savings)

    Returns: r
    '''
    A, gamma, epsilon, delta = params
    #r = gamma * A * ((L / K) ** (1 - gamma)) - delta
    y_params = np.array([A, gamma, epsilon])
    Y = get_Y(y_params, K, L)
    r = (A**((epsilon-1)/epsilon))*(((gamma*Y)/K)**(1/epsilon)) - delta
    return r


def get_w(params, K, L):
    '''
    Generates real wage w from parameters, aggregate capital stock K,
    and aggregate labor L

    Inputs:
        params = [2,] vector, [A, alpha]
        A      = scalar > 0, total factor productivity of production
                 function
        alpha  = scalar in (0,1), capital share of income in production
                 function
        K      = scalar > 0, aggregate capital stock
        L      = scalar > 0, aggregate labor

    Functions called: None

    Objects in function:
        w = scalar > 0, real wage

    Returns: w
    '''
    A, gamma, epsilon = params
    #w = (1 - gamma) * A * ((K / L) ** gamma)
    y_params = np.array([A, gamma, epsilon])
    Y = get_Y(y_params, K, L)
    w = (A**((epsilon-1)/epsilon))*((((1-gamma)*Y)/L)**(1/epsilon)) 
    return w






