'''
------------------------------------------------------------------------
Last updated 7/15/2015

This file contains functions for creating aggregate variables and checking
aggregate constraints.
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


def feasible(params, bvec):
    '''
    Determines whether a particular guess for the steady-state
    distribution of savings is feasible, first in terms of K>0, then in
    terms of c_s>0 for all s

    Inputs:
        params = [4,] vector, [S, A, alpha, delta]
        S      = integer in [3,80], number of periods in a life
        A      = scalar > 0, total factor productivity of production
                 function
        alpha  = scalar in (0,1), capital share of income in production
                 function
        delta  = scalar in [0,1], model-period depreciation rate of
                 capital
        bvec   = [S-1,] vector, initial guess for distribution of
                 savings b_{s+1}

    Functions called:
        get_K       = generates aggregate capital stock from bvec
        get_L       = generates aggregate labor from nvec
        get_r       = generates interest rate from r_params, K, and L
        get_w       = generates real wage from w_params, K, and L
        get_cvec_ss = generates consumption vector and c_constr from r,
                      w, and bvec

    Objects in function:
        GoodGuess = boolean, =True if initial steady-state guess is
                    feasible
        K         = scalar > 0, aggregate capital stock
        K_constr  = boolean, =True if K<=0 for given bvec
        c_constr  = [S,] boolean vector, =True if c<=0 for given bvec
        L         = scalar>0, aggregate labor
        r_params  = [3,] vector, parameters for r-function
                    [A, alpha, delta]
        r         = scalar > 0, interest rate (real return on savings)
        w_params  = [2,] vector, parameters for w-function [A, alpha]
        w         = scalar > 0, real wage
        cvec      = [S,] vector, consumption c_s for each age-s agent

    Returns: GoodGuess, K_constr, c_constr
    '''
    S, A, alpha, delta = params
    GoodGuess = True
    # Check K
    K, K_constr = get_K(bvec)
    if K_constr == True:
        GoodGuess = False
    # Check cvec if K has no problems
    c_constr = np.zeros(S, dtype=bool)
    if K_constr == False:
        L = get_L(np.ones(S))
        r_params = np.array([A, alpha, delta])
        r = get_r(r_params, K, L)
        w_params = np.array([A, alpha])
        w = get_w(w_params, K, L)
        cvec, c_constr = get_cvec_ss(S, r, w, bvec)
    if c_constr.max() == True:
        GoodGuess = False
    return GoodGuess, K_constr, c_constr



def get_K(bvec):
    '''
    Generates aggregate capital stock K from distribution of individual
    savings

    Inputs:
        bvec = [S-1,] vector, distribution of savings b_{s+1}

    Functions called: None

    Objects in function:
        K_constr = boolean, =True if K<=0 for given bvec
        K        = scalar, aggregate capital stock

    Returns: K, K_constr
    '''
    K_constr = False
    K = bvec.sum()
    if K <= 0:
        print 'b matrix and/or parameters resulted in K<=0'
        K_constr = True
    return K, K_constr



def get_C(cvec):
    '''
    Generates aggregate consumption C

    Inputs:
        cvec = [S,] vector, distribution of consumption c_s

    Functions called: None

    Objects in function:
        C = scalar > 0, aggregate consumption

    Returns: C
    '''
    C = cvec.sum()
    return C

def SS(params, b_guess, graphs):
    '''
    Generates all endogenous steady-state objects

    Inputs:
        params  = [7,] vector,
                  [S, beta, sigma, A, alpha, delta, SS_tol]
        S       = integer in [3,80], number of periods an individual
                  lives
        beta    = scalar in [0,1), discount factor
        sigma   = scalar > 0, coefficient of relative risk aversion
        A       = scalar > 0, total factor productivity parameter in
                  firms' production function
        alpha   = scalar in (0,1), capital share of income
        delta   = scalar in [0,1], model-period depreciation rate of
                  capital
        SS_tol  = scalar > 0, tolerance level for steady-state fsolve
        b_guess = [S-1,] vector, initial guess for the distribution
                  of savings b_{s+1}
        graphs  = boolean, =True if want graphs of steady-state objects

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
        eul_params   = [6,] vector, [S, beta, sigma, A, alpha, delta]
        b_ss         = [S-1,] vector, steady state distribution of
                       savings
        L_ss         = scalar > 0, steady-state aggregate labor
        K_ss         = scalar > 0, steady-state aggregate capital stock
        K_constr     = boolean, =True if K<=0 for given bvec
        r_params     = [3,] vector, parameters for r-function
                       [A, alpha, delta]
        r_ss         = scalar > 0, steady-state interest rate (real
                       return on savings)
        w_params     = [2,] vector, parameters for w-function
                       [A, alpha]
        w_ss         = scalar > 0, steady-state real wage
        c_ss         = [S,] vector, steady-state consumption c_s for
                       each age-s agent
        c_constr     = [S,] boolean vector, =True if c<=0 for given
                       bvec
        b_err_params = [3,] vector, parameters for Euler errors
                       [S, beta, sigma]
        EulErr_ss    = [S-1,] vector, vector of steady-state Euler
                       errors
        svec         = [S,] vector, age-s indices from 1 to S
        b_ss0        = [S,] vector, age-s wealth levels including b_1=0

    Returns: b_ss, c_ss, w_ss, r_ss, K_ss, EulErr_ss
    '''
    S, beta, sigma, alpha, cbar1, cbar2, A, gamma, epsilon, delta, SS_tol = params
    b_ss = opt.fsolve(EulerSys, b_guess, args=(params), xtol=SS_tol)
    
    # Generate other steady-state values and Euler equations
    L_ss = get_L(np.ones(S))
    K_ss, K_constr = get_K(b_ss)
    r_params = np.array([A, gamma, epsilon, delta])
     
    kl_params = np.array([A, gamma, epsilon, delta, SS_tol])
    K1_ss, L1_ss = get_KL(kl_params, L_ss, K_ss)
    K2_ss = K_ss - K1_ss
    L2_ss = L_ss - L1_ss
    y_params = np.array([A, gamma, epsilon])
    Y1_ss = get_Y(y_params, K1_ss, L1_ss)
    Y2_ss = get_Y(y_params, K2_ss, L2_ss)
    r_params = np.array([A, gamma, epsilon, delta])
    r_ss = get_r(r_params, K1_ss, L1_ss)
    w_params = np.array([A, epsilon, gamma])
    w_ss = get_w(w_params, K1_ss, L1_ss)
    p_params = np.array([A, gamma, epsilon, delta])
    p_c1_ss = get_p_c(p_params, r_ss, w_ss)
    p_c2_ss = get_p_c(p_params, r_ss, w_ss)
    p_tilde_ss = get_p_tilde(alpha, p_c1_ss, p_c2_ss)
    c_ss, c_constr = get_cvec_ss(S, r_ss, w_ss, b_ss, cbar1, cbar2, p_c1_ss, p_c2_ss, p_tilde_ss) # this gives composite consumption
    C_ss = get_C(c_ss)
    c1_ss = (p_tilde_ss*c_ss*alpha)/p_c1_ss + cbar1 # should make these functions and handle all goods at ones (array operations)
    c2_ss = (p_tilde_ss*c_ss*(1-alpha))/p_c2_ss + cbar2
    C1_ss = get_C(c1_ss)
    C2_ss = get_C(c2_ss)
    b_err_params = np.array([S, beta, sigma])
    EulErr_ss = get_b_errors(b_err_params, r_ss, c_ss, c_constr, diff=True)
      
    print('ss factor prices')
    print(r_ss)
    print(w_ss)  
    
    print('cons prices')
    print(p_c1_ss)
    print(p_c2_ss)
    print(p_tilde_ss)
    # if graphs == True:
    #     # Plot steady-state distribution of savings
    #     svec = np.linspace(1, S, S)
    #     b_ss0 = np.append([0], b_ss)
    #     minorLocator   = MultipleLocator(1)
    #     fig, ax = plt.subplots()
    #     plt.plot(svec, b_ss0)
    #     # for the minor ticks, use no labels; default NullFormatter
    #     ax.xaxis.set_minor_locator(minorLocator)
    #     plt.grid(b=True, which='major', color='0.65',linestyle='-')
    #     plt.title('Steady-state distribution of savings')
    #     plt.xlabel(r'Age $s$')
    #     plt.ylabel(r'Individual savings $\bar{b}_{s}$')
    #     plt.savefig('b_ss_Sec2')
    #     plt.show()

    #     # Plot steady-state distribution of consumption
    #     fig, ax = plt.subplots()
    #     plt.plot(svec, c_ss)
    #     # for the minor ticks, use no labels; default NullFormatter
    #     ax.xaxis.set_minor_locator(minorLocator)
    #     plt.grid(b=True, which='major', color='0.65',linestyle='-')
    #     plt.title('Steady-state distribution of consumption')
    #     plt.xlabel(r'Age $s$')
    #     plt.ylabel(r'Individual consumption $\bar{c}_{s}$')
    #     plt.savefig('c_ss_Sec2')
    #     plt.show()

    return b_ss, EulErr_ss, C1_ss, C2_ss, Y1_ss, Y2_ss, K_ss, K1_ss, K2_ss 
