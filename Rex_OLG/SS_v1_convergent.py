'''
Author: Rex McArthur
Last updated: 09/16/2015

Calculates Steady state OLG model with 80 age cohorts, 1 static firm

'''
#Packages
import numpy as np
import scipy.optimize as opt
import time

'''
Set up
______________________________________________________
S            = number of periods an individual lives
T            = number of time periods until steady state is reached
beta_annual  = discount factor for one year
beta         = discount factor for each age cohort
sigma        = coefficient of relative risk aversion
gamma        = capital share of income
epsilon      = elasticity of substitution between capital and labor
alpha        = Share of goods in composite consump
A            = total factor productivity parameter in firms' production
               function
lamb         = convex combo param
delta_annual = depreciation rate of capital for one year
delta        = depreciation rate of capital for each cohort
cbar       = minimum value amount of consumption
ltilde       = measure of time each individual is endowed with each
               period
chi_n        = discount factor of labor
chi_b        = discount factor of incidental bequests
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPImindist   = Cut-off distance between iterations for TPI
______________________________________________________
'''

# Parameters
sigma = 1.9 # coeff of relative risk aversion for hh
beta = 0.98 # discount rate
delta = .1
epsilon = 0.55
gamma = 0.4
lamb = .1
A = 1.0 # Total factor productivity
S = 80 # periods in life of hh

# Computational parameters
maxiter = 1000 #maximum iterations on convergent sol'n method
converge_tol = 1e-9 # tolerance for convergent sol'n method
mu = 0.1


def perc_dif_func(simul, data):
    '''
    Used to calculate the absolute percent difference between the data
    moments and model moments
    '''
    frac = (simul - data)/data
    output = np.abs(frac)
    return output


def consumption(w,r,n,b):
    '''
    Returns S length Consumption vector
    Eq. 11.45 in Rick's write up
    Params
    w - wage guess
    r - rate guess
    n - labor vector
    p - composite price
    b0 - kapital vector for first period
    b1 = capital bector for next period
    minimum - minimum bundle of good required
    '''
    b1 = np.zeros(S)
    b2 = np.zeros(S)
    b1[1:]  = b
    b2[:-1] = b
    c = ((1+r) * b1) + (w * n) - b2
    cmask = c < 0
    c[cmask] = 1e-15 # moved this here - no sense passing it to bvec
    return c


def mu_c(cvec):
    return cvec**(-sigma)

def get_cb(r, w, b_guess,nvec):
    '''
    Generates vectors for individual savings, composite consumption,
    industry specific consumption, constraint vectors, and Euler errors
    given r, w, p_comp, p_vec.
    '''


    bvec = opt.fsolve(savings_euler, b_guess, args =(r, w, nvec))
    cvec = consumption(w,r,nvec,bvec)
    eul_vec = get_b_errors(r, cvec)
    return bvec, cvec, eul_vec


def get_b_errors(r, cvec):
    '''

    '''

    mu_c0 = mu_c(cvec[:-1])
    mu_c1 = mu_c(cvec[1:])
    b_errors = (beta * (1+r)*mu_c1)-mu_c0
    #b_errors[cmask[:-1]] = 10e4
    #b_errors[cmask[1:]] = 10e4 DON'T NEED THESE IF FIX THE cvec[cmask] above...
    return b_errors

def savings_euler(b_guess, r, w, nvec):
    '''
    Calculates the steady state savings vector using an fsolve
    Eq. 11.43 and 11.44 in Rick's write up
    Returns a vector of S-1 euler errors to be used in an fsolve
    Params
    Savings_guess - an S-1 array of intial guesses for savings
    r - current rate guess
    w - current wage guess
    p - current composite price calculation
    minimum - minimum bundle needed by law
    '''
    c = consumption(w, r, nvec, b_guess)
    #print c
    b_error_vec = get_b_errors(r, c)
    #print 'b_error ', b_error_vec
    return b_error_vec


def calc_new_r(Y,K):
    '''
    Gives a new implied interest rate using FOC for a firm
    '''
    MPK = calc_MPK(Y,K)
    r_new = MPK-delta
    return r_new

def calc_new_w(Y, L):
    '''
    Gives a new implied wage rate using FOC for a firm
    '''
    w_new = calc_MPL(Y,L)
    return w_new


def calc_MPK(Y,K):
    '''
    Calculates the marginal product of capital
    '''
    MPK = (((gamma*Y)/K)**(1/epsilon))*A**((epsilon-1)/epsilon)
    return MPK

def calc_MPL(Y,L):
    '''
    Calculates the marginal product of labor
    '''
    MPL = ((((1-gamma)*Y)/L)**(1/epsilon))*(A**((epsilon-1)/epsilon))
    return MPL

def prod_func(K,L):
    '''
    Calculates the value of output from the production function
    '''
    Y = (A * (((gamma**(1/epsilon))*(K**((epsilon-1)/epsilon))) + 
          (((1-gamma)**(1/epsilon))*(L**((epsilon-1)/epsilon))))**(epsilon/(epsilon-1)))
    return Y


def rw_errors(rwvec, b_guess, nvec):
    '''
    Returns the capital and labor market clearing errors
    given r,w
    '''
    r,w = rwvec
    if r<= 0 or w<= 0:
        r_diff = 9999.
        w_diff = 9999.
    else:
        # solve household problem
        bvec, cvec, eulvec = \
            get_cb(r, w, b_guess, nvec)

        #Find aggregates
        C = cvec.sum()
        K = bvec.sum()
        L = nvec.sum()

        # Find implied factor prices
        # Using firm FOCs and market clearing
        Y = prod_func(K,L)
        rnew = calc_new_r(Y, K)
        wnew = calc_new_w(Y, L)
        #print 'new w', wnew
        #print 'new r', rnew
        r_diff = abs(rnew - r)
        if K <= 0: # check that aggregate savings is non-negative
            r_diff == 99999.
        w_diff = abs(wnew - w)

    rw_errors = np.array((r_diff, w_diff))
    return rw_errors


def ss_solve_fsolve(rw_init, b_guess, nvec):
    rw_ss = opt.fsolve(rw_errors, rw_init, args=(b_guess, nvec))
    r_ss, w_ss = rw_ss
    b_ss, c_ss, euler_error_ss = \
            get_cb(r_ss, w_ss, b_guess, nvec)

    # print resutlts to check
    #print 'b_ss: ', b_ss 
    #print 'c_ss: ', c_ss 
    #Find aggregates
    C_ss = c_ss.sum()
    K_ss = b_ss.sum()
    L_ss = nvec.sum()

    #print 'Aggregate results: ', C_ss, K_ss, L_ss

    # Find implied factor prices
    # Using firm FOCs and market clearing
    Y_ss = prod_func(K_ss,L_ss)
    rnew_ss = calc_new_r(Y_ss, K_ss)
    wnew_ss = calc_new_w(Y_ss, L_ss)

    r_diff_ss = abs(rnew_ss - r_ss)
    w_diff_ss = abs(wnew_ss - w_ss)

    SS_rw_errors = np.array((r_diff_ss, w_diff_ss))

    return (r_ss, w_ss, b_ss, c_ss, euler_error_ss,
            C_ss, K_ss, L_ss, Y_ss, SS_rw_errors)
 

def ss_solve_convex(rw_init,b_guess,nvec,mu):
    dist = 10
    iteration = 0 
    dist_vec = np.zeros(maxiter)
    r = rw_init[0]
    w = rw_init[1]
    while (dist > converge_tol) and (iteration < maxiter) :
        bvec, cvec, eulvec = \
            get_cb(r, w, b_guess, nvec)

        #Find aggregates
        C = cvec.sum()
        K = bvec.sum()
        L = nvec.sum()

        # Find implied factor prices
        # Using firm FOCs and market clearing
        Y = prod_func(K,L)
        rnew = calc_new_r(Y, K)
        wnew = calc_new_w(Y, L)

        print 'r, w: ', r, w
        print 'r_new, w_new: ', rnew, wnew
        
        r = mu*rnew + (1-mu)*r # so if r low, get low save, so low capital stock, so high mpk, so r_new bigger
        w = mu*wnew + (1-mu)*w

        dist = np.array([perc_dif_func(rnew, r)]+[perc_dif_func(wnew, w)]).max()
        
        dist_vec[iteration] = dist
        if iteration > 10:
            if dist_vec[iteration] - dist_vec[iteration-1] > 0:
                mu /= 2.0
                print 'New value of mu:', mu
        iteration += 1
        print "Iteration: %02d" % iteration, " Distance: ", dist

    return (r, w, bvec, cvec, eulvec,
            C, K, L, Y)


#nvec = np.array((1.,1.,.2))
nvec = np.ones(S)
rw_init = np.array(([0.05,.5]))
#bvec_guess = np.array((.1,.2))
bvec_guess = np.ones(S-1)*0.05

## Call fsolve method
#r_ss, w_ss, b_ss, c_ss, eul_ss, C_ss,\
#        K_ss, L_ss, Y_ss, SS_rw_errors = \
#        ss_solve_fsolve(rw_init, bvec_guess, nvec)

## Call convergent method
r_ss, w_ss, b_ss, c_ss, eul_ss, C_ss,\
        K_ss, L_ss, Y_ss = \
        ss_solve_convex(rw_init, bvec_guess, nvec,mu)
SS_rw_errors = 0

## Print SS output and model checks        
print 'ss r', r_ss
print 'ss w', w_ss


print 'rw errors', SS_rw_errors
print 'resource constraint check: ', Y_ss - C_ss - delta*K_ss
print 'euler errors', eul_ss
