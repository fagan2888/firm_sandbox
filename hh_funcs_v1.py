'''
------------------------------------------------------------------------
Last updated 7/15/2015

This file contains functions for the consumers problem.
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


def get_p_c(params, r, w):
    '''
    Generates price of consumption good/producer output

    Returns: p_c
    '''
    A, gamma, epsilon, delta = params
    p_c = (A**(epsilon-1))*(((1-gamma)*(w**(1-epsilon))) + (gamma*((r+delta)**(1-epsilon))))
    return p_c
    
def get_p_tilde(alpha, p_c1, p_c2):
    p_tilde = ((p_c1/alpha)**alpha)*((p_c2/(1-alpha))**(1-alpha))
    return p_tilde

def get_cvec_ss(S, r, w, bvec, cbar1, cbar2, p_c1, p_c2, p_tilde):
    '''
    Generates vector of consumptions from distribution of individual
    savings and the interest rate and the real wage

    Inputs:
        S    = integer in [3,80], number of periods an individual lives
        r    = scalar > 0, interest rate
        w    = scalar > 0, real wage
        bvec = [S-1,] vector, distribution of savings b_{s+1}.

    Functions called: None

    Objects in function:
        c_constr = [S,] boolean vector, =True if element c_s <= 0
        b_s      = [S,] vector, 0 in first element and bvec in last
                   S-1 elements
        b_sp1    = [S,] vector, bvec in first S-1 elements and 0 in
                   last element
        cvec     = [S,] vector, consumption by age c_s

    Returns: cvec, c_constr
    '''
    c_constr = np.zeros(S, dtype=bool)
    b_s = np.append([0], bvec)
    b_sp1 = np.append(bvec, [0])
    cvec = ((1 + r) * b_s + w - b_sp1 - (p_c1*cbar1) - (p_c2*cbar2))/p_tilde
    if cvec.min() <= 0:
        print 'initial guesses and/or parameters created c<=0 for some agent(s)'
        c_constr = cvec <= 0
    return cvec, c_constr


def get_b_errors(params, r, cvec, c_constr, diff):
    '''
    Generates vector of dynamic Euler errors that characterize the
    optimal lifetime savings

    Inputs:
        params   = [3,] vector, [p, beta, sigma]
        p        = integer in [2,80], remaining periods in life
        beta     = scalar in [0,1), discount factor
        sigma    = scalar > 0, coefficient of relative risk aversion
        r        = scalar > 0, interest rate
        cvec     = [p,] vector, distribution of consumption by age c_p
        c_constr = [p,] boolean vector, =True if c<=0 for given bvec
        diff     = boolean, =True if use simple difference Euler
                   errors. Use percent difference errors otherwise.

    Functions called: None

    Objects in function:
        mu_c     = [p-1,] vector, marginal utility of current
                   consumption
        mu_cp1   = [p-1,] vector, marginal utility of next period
                   consumption
        b_errors = [p-1,] vector, Euler errors with errors = 0
                   characterizing optimal savings bvec

    Returns: b_errors
    '''
    p, beta, sigma = params
    cvec[c_constr] = 9999. # Each consumption must be positive to
                           # generate marginal utilities
    mu_c = cvec[:p-1] ** (-sigma)
    mu_cp1 = cvec[1:] ** (-sigma)
    if diff == True:
        b_errors = (beta * (1 + r) * mu_cp1) - mu_c
        b_errors[c_constr[:p-1]] = 9999.
        b_errors[c_constr[1:]] = 9999.
    else:
        b_errors = ((beta * (1 + r) * mu_cp1) / mu_c) - 1
        b_errors[c_constr[:p-1]] = 9999. / 100
        b_errors[c_constr[1:]] = 9999. / 100
    return b_errors


def EulerSys(bvec, params):
    '''
    Generates vector of all Euler errors that characterize all
    optimal lifetime decisions

    Inputs:
        bvec   = [S-1,] vector, distribution of savings b_{s+1}
        params = [6,] vector, [S, beta, sigma, A, alpha, delta]
        S      = integer in [3,80], number of periods an individual
                 lives
        beta   = scalar in [0,1), discount factor
        sigma  = scalar > 0, coefficient of relative risk aversion
        A      = scalar > 0, total factor productivity parameter in firms'
                 production function
        alpha  = scalar in (0,1), capital share of income
        delta  = scalar in [0,1], model-period depreciation rate of
                 capital

    Functions called:
        get_L        = generates aggregate labor from nvec
        get_K        = generates aggregate capital stock from bvec
        get_r        = generates interest rate from r_params, K, and L
        get_w        = generates real wage from w_params, K, and L
        get_cvec_ss  = generates consumption vector and c_constr from
                       r, w, and bvec
        get_b_errors = generates vector of dynamic Euler errors that
                       characterize lifetime savings decisions

    Objects in function:
        L            = scalar > 0, aggregate labor
        K            = scalar > 0, aggregate capital stock
        K_constr     = boolean, =True if K<=0 for given bvec
        b_err_vec    = [S-1,] vector, vector of Euler errors
        r_params     = [3,] vector, parameters for r-function
                       [A, alpha, delta]
        r            = scalar > 0, interest rate (real return on
                       savings)
        w_params     = [2,] vector, parameters for w-function
                       [A, alpha]
        w            = scalar > 0, real wage
        cvec         = [S,] vector, consumption c_s for each age-s
                       agent
        c_constr     = [S,] boolean vector, =True if c<=0 for given
                       bvec
        b_err_params = [3,] vector, parameters for Euler errors
                       [S, beta, sigma]

    Returns: b_errors
    '''
    S, beta, sigma, alpha, cbar1, cbar2, A, gamma, epsilon, delta, SS_tol = params
    L = get_L(np.ones(S))
    K, K_constr = get_K(bvec)
    if K_constr == True:
        b_err_vec = 1000 * np.ones(S-1)
    else:
        kl_params = np.array([A, gamma, epsilon, delta, SS_tol])
        K1, L1 = get_KL(kl_params, L, K)
        K2 = K - K1
        L2 = L - L1
        r_params = np.array([A, gamma, epsilon, delta])
        r = get_r(r_params, K1, L1)
        w_params = np.array([A, gamma, epsilon])
        w = get_w(w_params, K1, L1)
        p_params = np.array([A, gamma, epsilon, delta])
        p_c1 = get_p_c(p_params, r, w)
        p_c2 = get_p_c(p_params, r, w)
        p_tilde = get_p_tilde(alpha, p_c1, p_c2)
        cvec, c_constr = get_cvec_ss(S, r, w, bvec, cbar1, cbar2, p_c1, p_c2, p_tilde)
        b_err_params = np.array([S, beta, sigma])
        b_err_vec = get_b_errors(b_err_params, r, cvec, c_constr,
                                 diff=True)
    return b_err_vec

