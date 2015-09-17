'''
Author: Rex McArthur
Last updated: 09/03/2015

Calculates Steady state OLG model with 3 age cohorts, 2 static firms

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
alpha = np.array([0.5,1-0.5]) # preference parameter - share of good i in composite consumption, shape =(I,), shares must sum to 1
cbar = np.array([0.00, 0.00]) # min cons of each of I goods, shape =(I,)
delta = .1
epsilon = np.array((.55, .45))
gamma = np.array((.5, .5))
lamb = .1
A = 1.0 # Total factor productivity
S = 80 # periods in life of hh
I = 2 # number of consumption goods
M = 2 # number of production industries
nvec = np.array((1.,1.,1.,.2, ))



def firm_price(r,w):
    '''
    Calculates the firm price
    Eq. 11.41 in Rick's write up
    Returns an M length vector of prices for each firm
    Params
    r - rate guess
    w - wage guess
    '''
    P =  (1/A)*(gamma*(r+delta)**(1-epsilon)+
            (1-gamma)*w**(1-epsilon))**(1/(1-epsilon))
    return P

def comp_price(price_vec):
    '''
    Assuming you have 2 firms, finds the price of the composite good
    Eq. 11.42 in Rick's write up
    Returns a composite price
    Params
    price_vec - a vector of the two prices
    '''

    return ((price_vec[0]/alpha[0])**alpha[0])*(price_vec[1]/(alpha[1]))**(alpha[1])

def min_consump(p):
    '''
    Calculate the minimum consumption necessary 
    Returns the value of the minimum consumption bundle
    params
    p - Price vector of goods
    '''
    return np.sum(cbar*p)

def consumption(w,r,n,p_vec,p_comp,b):
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
    minimum = min_consump(p_vec)
    c = (((1/p_comp) * ((1+r) * b1 + w * n - b2 - minimum)))
    cmask = c < 0
    return c, cmask


def mu_c(cvec):
    return cvec**sigma

def get_cb(r, w, b_guess, p_vec, p_comp, nvec):
    '''
    Generates vectors for individual savings, composite consumption,
    industry specific consumption, constraint vectors, and Euler errors
    given r, w, p_comp, p_vec.
    '''

    bvec = opt.fsolve(savings_euler, b_guess, args =(r, w, p_comp, p_vec, nvec))
    cvec, c_cstr = consumption(w,r,nvec, p_vec, p_comp, bvec)
    cm_opt, cm_cstr = get_cm_optimal(cvec, p_comp, p_vec)
    eul_vec = get_b_errors(r, cvec, c_cstr)
    return bvec, cvec, c_cstr, cm_opt, cm_cstr, eul_vec

def get_cm_optimal(cvec, p_comp, p_vec):
    c1vec = ((alpha[0] * p_comp * cvec)/p_vec[0])+ cbar[0]
    c2vec = ((alpha[1] * p_comp * cvec)/p_vec[1])+ cbar[1]
    cm_opt = np.vstack((c1vec, c2vec)) 
    cm_cstr = cm_opt <= 0
    return cm_opt, cm_cstr


def get_b_errors(r, cvec, cmask):
    '''

    '''
    cvec[cmask] = 9999.
    mu_c0 = mu_c(cvec[:-1])
    mu_c1 = mu_c(cvec[1:])
    b_errors = (beta * (1+r)*mu_c0)-mu_c1
    b_errors[cmask[:-1]] = 10e4
    b_errors[cmask[1:]] = 10e4
    return b_errors

def savings_euler(b_guess, r, w, p_comp, p_vec, minimum):
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
    c, c_cstr = consumption(w, r, nvec, p_vec, p_comp, b_guess)
    print c
    b_error_vec = get_b_errors(r, c, c_cstr)
    print 'b_error ', b_error_vec
    return b_error_vec

def hh_ss_consumption(c, cum_p, prices):
    '''

    '''
    ss_good_1 = alpha[0]*((cum_p*c)/prices[0]+cbar[0])
    ss_good_2 = alpha[1]*((cum_p*c)/prices[1]+cbar[1])
    return np.array([ss_good_1, ss_good_2])

def agg_consump(consump_demand):
    agg_demand = []
    for i in xrange(len(consump_demand)):
        agg_demand.append(np.sum(consump_demand[i]))
    return agg_demand

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

def get_K(Y, pbar, com_p, r, w):
    '''
    Returns the Capital Demand of the firms
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

def get_L(K, r, w):
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

def calc_new_r(p, Y, K):
    '''
    Gives a new implied interest rate using FOC for a firm
    '''
    r_new = p*((gamma[0]*Y)/K)*A**((epsilon[0]-1)/1)-delta
    return r_new

def calc_new_w(p, Y, L):
    '''
    Gives a new implied wage rate using FOC for a firm
    '''
    w_new = p*(((1-gamma[0])*Y)/L)*A**((epsilon[0]-1)/1)
    return w_new

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

def market_errors(rwvec, b_guess, nvec):
    '''
    Returns the capital and labor market clearing errors
    given r,w
    '''
    r,w = rwvec
    if r+delta <=0 or w<= 0:
        r_diff = 9999.
        w_diff = 9999.
    else:
        p_vec = firm_price(r,w)
        p_comp = comp_price(p_vec)
        bvec, cvec, c_cstr, cm_opt, c_opt_cstr, eulvec = \
            get_cb(r, w, b_guess, p_vec, p_comp, nvec)

        #Problem here with the bvec
        C_demand = cm_opt.sum(axis = 1)
        Y_m = get_Y(C_demand, r, w)
        Y_total = np.sum(Y_m)
        K_demand = get_K(Y_m, p_vec, p_comp, r, w)
        L_demand = get_L(K_demand, r, w)
        K_supply = np.sum(bvec)
        L_supply = np.sum(nvec)
        l_res = calc_k_res(K_supply, K_demand)
        k_res = calc_l_res(L_supply, L_demand)
        rnew = calc_new_r(p_comp, Y_total, k_res)
        wnew = calc_new_w(p_comp, Y_total, l_res)
        print 'new w', wnew
        print 'new r', rnew
        r_diff = abs(rnew - r)
        w_diff = abs(wnew - w)

   
    C_demand_ss = cm_opt_ss.sum(axis = 1)
    Y_m_ss = get_Y(C_demand_ss, r_ss, w_ss)
    K_demand_ss = get_K(Y_m_ss, prices_ss, com_price_ss, r_ss, w_ss)
    L_demand_ss = get_L(K_demand_ss, r_ss, w_ss)
    k_error_ss = np.sum(K_demand_ss) - np.sum(b_ss)
    l_error_ss = np.sum(L_demand_ss) - np.sum(nvec)
    SS_market_errors = np.array([k_error_ss, l_error_ss])


        
    market_errors = np.array((r_diff, w_diff))
    return market_errors

def ss_solve_convex(rw_init,nvec):
    error = 1
    r_guess = rw_init[0]
    w_guess = rw_init[1]
    while error > .000001:
        prices = firm_price(r_guess,w_guess)
        minimum = min_consump(prices)
        com_price = comp_price(prices)
        guessvec = np.ones(S-1)*0.01
        newb = opt.fsolve(savings_euler, guessvec, args = (r_guess, w_guess, com_price, minimum))
        c_guess, cmask = consumption(w_guess, r_guess, nvec, com_price, newb, minimum)
        ss_consump = hh_ss_consumption(c_guess, com_price, prices)
        #print ss_consump
        Cbar = agg_consump(ss_consump)
        Ybar = get_Y(Cbar, r_guess, w_guess)
        K_demand = get_K(Ybar, prices, com_price, r_guess, w_guess)
        L_demand = get_L(K_demand, r_guess, w_guess)
        L_mkt_clear = lab_clear(L_demand, nvec)
        K_mkt_clear = cap_clear(K_demand, newb)
        print 'Market Clearing conditions '
        print 'Labor: ', L_mkt_clear
        print 'Capital: ', K_mkt_clear
        #print 'prices: {}, Consumption: {}, K: {}'.format(prices, Ybar, K_demand)
        r_new = calc_new_r(prices[0], Ybar[0], K_demand[0])
        w_new = calc_new_w(prices[0], Ybar[0], L_demand[0])
        #print 'w', w_new
        #print 'r', r_new
        diff = np.array(([abs(r_guess-r_new),abs(w_guess-w_new)]))
        error = np.max(diff)
        print error, '\n'
        r_guess = lamb*r_new + (1-lamb)*r_guess
        w_guess = lamb*w_new + (1-lamb)*w_guess
    return np.array(([r_new, w_new]))
        

def ss_solve_fsolve(rw_init, b_guess, nvec):
    rw_ss = opt.fsolve(market_errors, rw_init, args=(b_guess, nvec))
    r_ss, w_ss = rw_ss
    prices_ss = firm_price(r_ss,w_ss)
    minimum_ss = min_consump(prices_ss)
    com_price_ss = comp_price(prices_ss)
    b_ss, c_ss, c_cstr, cm_opt_ss, cm_opt_cstr, euler_error_ss = \
            get_cb(r_ss, w_ss, b_guess, prices_ss, com_price_ss, nvec)
    C_demand_ss = cm_opt_ss.sum(axis = 1)
    Y_m_ss = get_Y(C_demand_ss, r_ss, w_ss)
    K_demand_ss = get_K(Y_m_ss, prices_ss, com_price_ss, r_ss, w_ss)
    L_demand_ss = get_L(K_demand_ss, r_ss, w_ss)
    print L_demand_ss
    k_res = calc_k_res(np.sum(b_ss), K_demand_ss)
    l_res = calc_l_res(np.sum(nvec), L_demand_ss)
    k_error_ss = np.sum(K_demand_ss) - np.sum(b_ss)
    l_error_ss = np.sum(L_demand_ss) - np.sum(nvec)
    SS_market_errors = np.array([k_error_ss, l_error_ss])
    print 'resource constraint', np.sum(Y_m_ss)


    return (r_ss, w_ss, prices_ss, com_price_ss, b_ss, c_ss, cm_opt_ss, euler_error_ss,
            C_demand_ss, Y_m_ss, K_demand_ss, L_demand_ss, SS_market_errors)
    

#nvec = np.array((1.,1.,.2))
nvec = np.ones(S)
rw_init = np.array(([0.05,.5]))
#bvec_guess = np.array((.1,.2))
bvec_guess = np.ones(S)*0.01
r_ss, w_ss, prices_ss, com_p_ss, b_ss, c_ss, cm_ss, eul_ss, C_demand_ss,\
        Y_m_ss, K_demand_ss, L_demand_ss, SS_market_errors = \
        ss_solve_fsolve(rw_init, bvec_guess, nvec)
print 'ss r', r_ss
print 'ss w', w_ss
print 'ss prices', prices_ss
print 'composite p', com_p_ss
print 'b_ss', b_ss
print 'c', c_ss
print 'cm', cm_ss
print 'market errors', SS_market_errors
print 'euler errors', eul_ss
