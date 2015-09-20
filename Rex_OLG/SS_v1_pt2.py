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
alpha = .6
delta = .1
epsilon = .55 
gamma = np.array([0.4, .5])
lamb = .1
cbar = np.array([0.00,0.00])
A = 1.0 # Total factor productivity
S = 80 # periods in life of hh
I = 2
M = 2

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

def calc_price_vec(r,w):
    '''
    Calculates the firm price
    Eq. 11.41 in Rick's write up
    Returns an M length vector of prices for each firm
    Params
    r - rate guess
    w - wage guess
    '''
    Pvec =  (1/A)*(gamma*(r+delta)**(1-epsilon)+
            (1-gamma)*w**(1-epsilon))**(1/(1-epsilon))
    return Pvec

def comp_price(price_vec):
    '''
    Assuming you have 2 firms, finds the price of the composite good
    Eq. 11.42 in Rick's write up
    Returns a composite price
    Params
    price_vec - a vector of the two prices
    '''

    return ((price_vec[0]/alpha)**alpha)*(price_vec[1]/(1-alpha))**(1-alpha)

def min_consump(p):
    '''
    Calculate the minimum consumption necessary 
    Returns the value of the minimum consumption bundle
    params
    p - Price vector of goods
    '''
    return np.sum(cbar*p)

def consumption(w, r, n, b, p_m):
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
    minimum = min_consump(p_m)
    c = ((1+r) * b1) + (w * n) - b2 - minimum
    cmask = c < 0
    if cmask.any() == True:
        print c
        print 'C error'
    c[cmask] = 1e-15 
    return c

def get_cmmat(cvec, pmvec, cmtilvec, p):
    '''
    Generates matrix of consumptions of each type of good given prices
    and composite consumption
    Inputs:
    p - composite price

    Returns: cmmat
    '''
    c1vec = ((alpha * p * cvec) / pmvec[0]) + cmtilvec[0]
    c2vec = (((1 - alpha) * p * cvec) / pmvec[1]) + cmtilvec[1]
    cmmat = np.vstack((c1vec, c2vec))
    cm_cstr = cmmat <= 0
    #cmmat[cm_cstr] = 1e-4
    return cmmat, cm_cstr

def get_cb(r, w, b_guess,nvec, p_comp):
    '''
    Generates vectors for individual savings, composite consumption,
    industry specific consumption, constraint vectors, and Euler errors
    given r, w, p_comp, p_vec.
    '''
    p_m = calc_price_vec(r,w)
    bvec = opt.fsolve(savings_euler, b_guess, args =(r, w, nvec, p_m))
    cvec = consumption(w,r,nvec,bvec, p_m)
    cmmat, cm_cstr = get_cmmat(cvec, p_m, cbar, p_comp)
    eul_vec = get_b_errors(r, cvec)
    return bvec, cvec, cmmat, cm_cstr, eul_vec


def get_b_errors(r, cvec):
    '''

    '''
    mu_c0 = (cvec[:-1])**(-sigma)
    mu_c1 = (cvec[1:])**(-sigma)
    b_errors = (beta * (1+r)*mu_c1)-mu_c0
    return b_errors

def savings_euler(b_guess, r, w, nvec,p_m):
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
    c = consumption(w, r, nvec, b_guess, p_m)
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
    
    MPK = (((gamma[0]*Y)/K)**(1/epsilon))*A**((epsilon-1)/epsilon)
    return MPK

def calc_MPL(Y,L):
    '''
    Calculates the marginal product of labor
    '''
    MPL = ((((1-gamma[0])*Y)/L)**(1/epsilon))*(A**((epsilon-1)/epsilon))
    return MPL

def prod_func(K,L):
    '''
    Calculates the value of output from the production function
    '''
    Y_m = (A * (((gamma**(1/epsilon))*(K**((epsilon-1)/epsilon))) + 
          (((1-gamma)**(1/epsilon))*(L**((epsilon-1)/epsilon))))**(epsilon/(epsilon-1)))
    return Y_m

def get_K_demand(Y, pbar, com_p, r, w):
    '''
    Returns (M) vector of firms capital demand
    Params
    r - current rate guess
    w - current wage guess
    Y - Output of each firm, M firms
    pbar - price of ecah good, M prices
    com_P - weighted composite price p
    '''
    K = (Y/A)*(gamma**(1/epsilon)+(1-gamma)**(1/epsilon)*((r+delta)/w)**
            (epsilon-1)*((1-gamma)/gamma)**((epsilon-1)/epsilon))**(epsilon/(1-epsilon))
    return K

def get_L_demand(K, r, w):
    '''
    Returns the Labor Demand of the firms
    Params
    r - current rate guess
    w - current wage guess
    Y - Output of each firm, M firms
    pbar - price of ecah good, M prices
    com_P - weighted composite price p
    '''
    L = K*((1-gamma)/gamma)*((r+delta)/w)**epsilon
    return L

def get_Y(C, r, w):
    '''
    Returns an M length vector of the output for each firm
    Parameters
    r - current rate guess
    w - current wage guess
    C - Consumption of each good M length vector
    '''
    Y = C*(1-(delta/A)*(gamma**(1/epsilon)+(1-gamma)**(1/epsilon)*
        ((r+delta)/w)**(epsilon-1)*((1-gamma)/gamma)**((epsilon-1)/epsilon))**
        (epsilon/(1-epsilon)))**(-1)
    return Y

def calc_k_res(k_supply, k_demand):
    k_res = k_supply - np.sum(k_demand[1:])
    if k_res <0:
        k_res = .00001
    return k_res

def calc_l_res(l_supply, l_demand):
    l_res = l_supply - np.sum(l_demand[1:])
    if l_res <0:
        l_res = .00001
    return l_res


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
        p_m = calc_price_vec(r,w)
        p_comp = comp_price(p_m)
        bvec, cvec, cm_mat, cm_cstr, eulvec = \
            get_cb(r, w, b_guess, nvec, p_comp)

        #Find aggregates
        C_m = cm_mat.sum()
        K_supply = bvec.sum()
        L_supply = nvec.sum()

        #find factor demand
        Y_m = get_Y(C_m, K_supply, L_supply)
        Km_demand = get_K_demand(Y_m, p_m, p_comp, r, w)
        Lm_demand = get_L_demand(Km_demand, r, w)
        #Calculate Residuals for industries
        k_res = calc_k_res(K_supply, Km_demand)
        l_res = calc_l_res(L_supply, Lm_demand)
        # Using firm FOCs and market clearing
        rnew = calc_new_r(Y_m[0], k_res)
        wnew = calc_new_w(Y_m[0], l_res)
        #print 'new w', wnew
        #print 'new r', rnew
        r_diff = abs(rnew - r)
        if K_supply <= 0: # check that aggregate savings is non-negative
            r_diff == 99999.
        w_diff = abs(wnew - w)

    rw_errors = np.array((r_diff, w_diff))
    return rw_errors


def ss_solve_fsolve(rw_init, b_guess, nvec):
    rw_ss = opt.fsolve(rw_errors, rw_init, args=(b_guess, nvec))
    r_ss, w_ss = rw_ss
    p_m_ss = calc_price_vec(r_ss,w_ss)
    p_comp_ss = comp_price(p_m_ss)

    b_ss, cm_ss, cm_mat_ss, cm_cstr, euler_error_ss = \
            get_cb(r_ss, w_ss, b_guess, nvec, p_comp_ss)

    # print resutlts to check
    #print 'b_ss: ', b_ss 
    #print 'c_ss: ', c_ss 
    #Find aggregates
    C_m_ss = cm_mat_ss.sum(axis = 1)
    K_supply_ss = b_ss.sum()
    L_supply_ss = nvec.sum()
    C_ss = C_m_ss.sum()

    #print 'Aggregate results: ', C_ss, K_ss, L_ss

    # Find implied factor prices
    # Using firm FOCs and market clearing
    Y_m_ss = get_Y(C_m_ss, K_supply_ss, L_supply_ss)
    Km_demand_ss = get_K_demand(Y_m_ss, p_m_ss, p_comp_ss, r_ss, w_ss)
    Lm_demand_ss = get_L_demand(Km_demand_ss, r_ss, w_ss)
    k_res = calc_k_res(K_supply_ss, Km_demand_ss)
    l_res = calc_l_res(L_supply_ss, Lm_demand_ss)
    rnew_ss = calc_new_r(Y_m_ss[0], k_res)
    wnew_ss = calc_new_w(Y_m_ss[0], l_res)

    r_diff_ss = abs(rnew_ss - r_ss)
    w_diff_ss = abs(wnew_ss - w_ss)

    SS_rw_errors = np.array((r_diff_ss, w_diff_ss))
    print K_supply_ss
    print L_supply_ss
    Y_ss = sum(prod_func(K_supply_ss, L_supply_ss))
    print 'Y', Y_ss

    return (r_ss, w_ss, b_ss, cm_ss, euler_error_ss,
            C_ss, K_supply_ss, L_supply_ss, Y_ss, SS_rw_errors)
 

#nvec = np.array((1.,1.,.2))
nvec = np.ones(S)
rw_init = np.array(([0.05,.5]))
#bvec_guess = np.array((.1,.2))
bvec_guess = np.ones(S-1)*0.05

## Call fsolve method
r_ss, w_ss, b_ss, c_ss, eul_ss, C_ss,\
        K_ss, L_ss, Y_ss, SS_rw_errors = \
        ss_solve_fsolve(rw_init, bvec_guess, nvec)

print 

## Call convergent method
#r_ss, w_ss, b_ss, c_ss, eul_ss, C_ss,\
#        K_ss, L_ss, Y_ss = \
#        ss_solve_convex(rw_init, bvec_guess, nvec,mu)
SS_rw_errors = 0

## Print SS output and model checks        
print 'ss r', r_ss
print 'ss w', w_ss
print 'b', b_ss
print 'c', c_ss


print 'rw errors', SS_rw_errors
print 'Y', Y_ss
print 'C', C_ss 
print 'd', delta
print 'K', K_ss
print 'resource constraint check: ', Y_ss - C_ss - delta*K_ss
print 'euler errors', eul_ss
