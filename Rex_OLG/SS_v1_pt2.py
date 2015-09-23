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
beta         = discount factor for each age cohort
sigma        = coefficient of relative risk aversion
gamma        = capital share of income
epsilon      = elasticity of substitution between capital and labor
alpha        = Share of goods in composite consump
A            = total factor productivity parameter in firms' production
               function
delta        = depreciation rate of capital for each cohort
cbar         = minimum value amount of consumption
f_tol        = the minimum f_solve tolerence allowed
I            = Number of firms
______________________________________________________
'''

# Parameters
sigma = 1.9 # coeff of relative risk aversion for hh
beta = 0.98 # discount rate
alpha = .6
delta = .05
epsilon = .55 
gamma = np.array([0.4, .5])
lamb = .1
cbar = np.array([0.00,0.00])
A = 1.0 # Total factor productivity
S = 80 # periods in life of hh
I = 2
f_tol = 1e-13


def calc_price_vec(r,w):
    '''
    Calculates the firm price
    Eq. 11.41 in Rick's write up
    Returns an M length vector of prices for each firm
    Params
    r - rate guess
    w - wage guess
    '''
    pmvec = (1 / A) * ((gamma * ((r + delta) ** (1 - epsilon)) +
            (1 - gamma) * (w ** (1 - epsilon))) ** (1 / (1 - epsilon)))
    return pmvec

def comp_price(price_vec):
    '''
    Assuming you have 2 firms, finds the price of the composite good
    Eq. 11.42 in Rick's write up
    Returns a composite price
    Params
    price_vec - a vector of the two prices
    '''
    p1, p2 = price_vec
    p_comp = (((p1 / alpha) ** alpha) *
        ((p2 / (1 - alpha)) ** (1 - alpha)))
    return p_comp

def get_consumption(r, w, nvec, bvec, p_m, cbar, p):
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
    b_s = np.append([0], bvec)
    b_sp1 = np.append(bvec, [0])
    cvec = (1 / p) * ((1 + r) * b_s + w * nvec -
           (p_m * cbar).sum() - b_sp1)
    c_cstr = cvec <= 0
    return cvec, c_cstr

def get_cmmat(cvec, pmvec, cbars, p):
    '''
    Generates matrix of consumptions of each type of good given prices
    and composite consumption
    Inputs:
    pmvec - price vector of each good
    cvec - consumption in each period
    cm

    Returns: cmmat
    '''
    c1vec = ((alpha * p * cvec) / pmvec[0]) + cbars[0]
    c2vec = (((1 - alpha) * p * cvec) / pmvec[1]) + cbars[1]
    cmmat = np.vstack((c1vec, c2vec))
    cm_cstr = cmmat <= 0
    return cmmat, cm_cstr

def get_cb(r, w, p_vec, b_guess,nvec, p_comp):
    '''
    Generates vectors for individual savings, composite consumption,
    industry specific consumption, constraint vectors, and Euler errors
    given r, w, p_comp, p_vec.
    '''
    bvec = opt.fsolve(savings_euler, b_guess, args=(r, w, p_comp, p_vec), xtol = f_tol)
    cvec, c_cstr = get_consumption(r, w, nvec, bvec, p_vec, cbar, p_comp)
    cmmat, cm_cstr = get_cmmat(cvec, p_vec, cbar, p_comp)
    eulvec = get_b_errors(r, cvec, c_cstr)
    return bvec, cvec, c_cstr, cmmat, cm_cstr, eulvec


def get_b_errors(r, cvec, c_cstr):
    '''

    '''
    cvec[c_cstr] = 1.
    mu_c = cvec[:-1] ** (-sigma)
    mu_cp1 = cvec[1:] ** (-sigma)
    b_errors = (beta * (1 + r) * mu_cp1) - mu_c
    b_errors[c_cstr[:-1]] = 9999.
    b_errors[c_cstr[1:]] = 9999.
    return b_errors

def savings_euler(b_guess, r, w, p_comp, p_vec):
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
    cvec, c_str = get_consumption(r, w, nvec, b_guess, p_vec, cbar, p_comp)
    b_error_vec = get_b_errors(r,cvec, c_str)
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

def get_YKmvec(r,w,Cmvec, pmvec):
    '''
    Generates vector of aggregate output Y_m of good m and capital
    demand K_m for good m given r and w

    Inputs:
        params = length 2 tuple, (r, w)
        r      = scalar > 0, interest rate
        w      = scalar > 0, real wage
        Cmvec  = [2,] vector, aggregate consumption of all goods
        pmvec  = [2,] vector, prices in each industry
        Avec   = [2,] vector, total factor productivity values for all
                 industries
        gamvec = [2,] vector, capital shares of income for all
                 industries
        epsvec = [2,] vector, elasticities of substitution between
                 capital and labor for all industries
        delvec = [2,] vector, model period depreciation rates for all
                 industries

    Functions called: None

    Objects in function:
        aa    = [2,] vector, gamvec
        bb    = [2,] vector, 1 - gamvec
        cc    = [2,] vector, (1 - gamvec) / gamvec
        dd    = [2,] vector, (r + delvec) / w
        ee    = [2,] vector, 1 / epsvec
        ff    = [2,] vector, (epsvec - 1) / epsvec
        gg    = [2,] vector, epsvec - 1
        hh    = [2,] vector, epsvec / (1 - epsvec)
        ii    = [2,] vector, ((1 / Avec) * (((aa ** ee) + (bb ** ee) *
                (cc ** ff) * (dd ** gg)) ** hh))
        Ymvec = [2,] vector, aggregate output of all industries
        Kmvec = [2,] vector, capital demand of all industries

    Returns: Ymvec, Kmvec
    '''
    aa = gamma
    bb = 1 - gamma
    cc = (1 - gamma) / gamma
    dd = (r + delta) / w
    ee = 1 / epsilon
    ff = (epsilon - 1) / epsilon
    gg = epsilon - 1
    hh = epsilon / (1 - epsilon)
    ii = ((1 / A) *
         (((aa ** ee) + (bb ** ee) * (cc ** ff) * (dd ** gg)) ** hh))
    Ymvec = Cmvec / (1 - delta * ii)
    Kmvec = Ymvec * ii
    return Ymvec, Kmvec

def calc_k_res(k_supply, k_demand):
    '''
    Calculates the difference between the Capital supply, and the capital 
    demanded by all firms but one. Assigns this as a residual value, to calculate
    a new implied w, r.
    Inputs:
    k_supply, (scaler) Total amount of capital supplied
    k_demand, (vector) Capital demanded by each firm
    returns residual capital
    '''
    k_res = k_supply - np.sum(k_demand[1:])
    if k_res <0:
        k_res = .00001
    return k_res

def calc_l_res(l_supply, l_demand):
    '''
    Calculates the difference between the labor supply, and the labor
    demanded by all firms but one. Assigns this as a residual value, 
    to calculate a new implied w, r.
    Inputs:
    l_supply, (scaler) Total amount of labor supplied
    l_demand, (vector) Labor demanded by each firm
    returns residual labor
    '''
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
        bvec, cvec, c_cstr, cm_mat, cm_cstr, eulvec = \
            get_cb(r, w, p_m, b_guess, nvec, p_comp)

        #Find aggregates
        C_m = cm_mat.sum(axis=1)
        K_supply = bvec.sum()
        L_supply = nvec.sum()

        #find factor demand
        Y_m, Km_demand = get_YKmvec(r,w,C_m, p_m)
        #Y_m = get_Y(C_m, K_supply, L_supply)

        #Km_demand = get_K_demand(Y_m, p_m, p_comp, r, w)
        Lm_demand = get_L_demand(Km_demand, r, w)
        #Calculate Residuals for industries
        k_res = calc_k_res(K_supply, Km_demand)
        l_res = calc_l_res(L_supply, Lm_demand)
        # Using firm FOCs and market clearing
        rnew = calc_new_r(Y_m[0], k_res)
        wnew = calc_new_w(Y_m[0], l_res)
        r_diff = abs(rnew - r)
        if Km_demand.any() <= 0: # check that aggregate savings is non-negative
            r_diff == 99999.
        w_diff = abs(wnew - w)

    rw_errors = np.array((r_diff, w_diff))
    return rw_errors


def ss_solve_fsolve(rw_init, b_guess, nvec):
    '''
    Solves the household problem, or rather finds the steady state r, w, 
    and uses those to find all other interesting variables.
    '''
    rw_ss = opt.fsolve(rw_errors, rw_init, args=(b_guess, nvec), xtol=f_tol)
    r_ss, w_ss = rw_ss
    p_m_ss = calc_price_vec(r_ss,w_ss)
    p_comp_ss = comp_price(p_m_ss)

    b_ss, cm_ss, cm_cstr_ss, cm_mat_ss, cm_mat_cstr, euler_error_ss = \
            get_cb(r_ss, w_ss, p_m_ss, b_guess, nvec, p_comp_ss)

    #Find all aggregates
    C_m_ss = cm_mat_ss.sum(axis = 1)
    K_supply_ss = b_ss.sum()
    L_supply_ss = nvec.sum()
    C_ss = C_m_ss.sum()

    Y_m_ss, Km_demand_ss = get_YKmvec(r_ss,w_ss,C_m_ss, p_m_ss)
    Lm_demand_ss = get_L_demand(Km_demand_ss, r_ss, w_ss)

    #Calculate residuals, and implied r, w
    k_res = calc_k_res(K_supply_ss, Km_demand_ss)
    l_res = calc_l_res(L_supply_ss, Lm_demand_ss)
    rnew_ss = calc_new_r(Y_m_ss[0], k_res)
    wnew_ss = calc_new_w(Y_m_ss[0], l_res)

    r_diff_ss = abs(rnew_ss - r_ss)
    w_diff_ss = abs(wnew_ss - w_ss)

    SS_rw_errors = np.array((r_diff_ss, w_diff_ss))

    return (r_ss, w_ss, b_ss, cm_ss, euler_error_ss,
            C_m_ss, Km_demand_ss, Lm_demand_ss, Y_m_ss, SS_rw_errors)
 

#Make intial guesses
nvec = np.ones(S)
rw_init = np.array(([0.03,2.2]))
bvec_guess = np.ones(S-1)*0.05

## Call fsolve method
r_ss, w_ss, b_ss, c_ss, eul_ss, C_ss,\
        K_ss, L_ss, Y_ss, SS_rw_errors = \
        ss_solve_fsolve(rw_init, bvec_guess, nvec)

## Print SS output and model checks        
print 'ss r', r_ss
print 'ss w', w_ss
print 'b\n', b_ss
print 'c\n', c_ss

print '\nY for each firm\n', Y_ss
print '\nConsumption for each firm\n', C_ss 
print '\nd\n', delta
print '\nCapital demand for each firm\n', K_ss
print '\nresource constraint check: ', Y_ss - C_ss - delta*K_ss
print 'euler errors', eul_ss
print '\nMax Euler error: ', np.amax(eul_ss)
