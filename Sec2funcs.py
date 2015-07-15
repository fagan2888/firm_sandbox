'''
------------------------------------------------------------------------
Last updated 7/8/2015

All the functions for the SS and TPI computation from Section 2.
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


def get_L(nvec):
    '''
    Generates aggregate labor L from distribution of individual labor
    supply

    Inputs:
        nvec = [S,] vector, distribution of labor supply n_s

    Functions called: None

    Objects in function:
        L = scalar, aggregate labor

    Returns: L
    '''
    L = nvec.sum()
    return L


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
    c1_ss = c_ss*alpha + cbar1 # should make these functions and handle all goods at ones (array operations)
    c2_ss = c_ss*(1-alpha) + cbar2
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
