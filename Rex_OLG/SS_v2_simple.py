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
cbar = np.array([0.001, 0.001]) # min cons of each of I goods, shape =(I,)
delta = .05
epsilon = np.array((.55, .45))
gamma = np.array((.6, .5))
lamb = .1
A = 1.0 # Total factor productivity
S = 3 # periods in life of hh
I = 2 # number of consumption goods
M = 2 # number of production industries


def lab_clear(L_demand, nvec):
    '''
    Checks the Capital market to see if it is cleared
    Eq. 11.36 in Rick's write-up
    Params
    L_demand - Labor demand for firms
    nvec - Labor supplied vector
    '''
    return np.sum(L_demand) - np.sum(nvec)

def cap_clear(K_demand, bvec):
    '''
    Checks the Labor market to see if it is cleared
    Eq. 11.35 in Rick's write-up
    Params
    K_demand - capital demand from firms
    bvec - capital supplied vector from households
    '''
    return np.sum(K_demand) - np.sum(bvec)

def firm_price(r,w):
    '''
    Calculates the firm price
    Eq. 11.41 in Rick's write up
    Returns an M length vector of prices for each firm
    Params
    r - rate guess
    w - wage guess
    '''
    return (1/A)*(gamma*(r+delta)**(1-epsilon)+
            (1-gamma)*w**(1-epsilon))**(1/(1-epsilon))

def comp_price(price_vec):
    '''
    Assuming you have 2 firms, finds the price of the composite good
    Eq. 11.42 in Rick's write up
    Returns a composite price
    Params
    price_vec - a vector of the two prices
    '''
    #TODO Change this to work for more than two firms
    return ((price_vec[0]/alpha[0])**alpha[0])*(price_vec[1]/(alpha[1]))**(alpha[1])

def min_consump(p):
    '''
    Calculate the minimum consumption necessary 
    Returns the value of the minimum consumption bundle
    params
    p - Price vector of goods
    '''
    return np.sum(cbar*p)

def consumption(w,r,n,p,b,minimum):
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
    b3 = np.zeros(S)
    b1[1:]  = b
    b2[:-1] = b
    b3[:-2] = b[1:]
    c = (((1/p) * ((1+r) * b1 + w * n - b2 - minimum)))
    print c
    cmask = c < 0
    if cmask.any() == True:
        print 'ERROR: consumption < 0'
    return c, cmask

def foc_k(r,c):
    '''
    Returns the first order condition vector for capital
    Params
    w - wage guess
    c - consumption vector
    '''
    error = c[:-1]**-sigma-(1+r)*beta*(c[1:])**-sigma
    return error

def savings_euler(savings_guess, r, w, p, minimum):
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
    #TODO Make this work with a bequest motive, so they can save in the last period
    #Also make it work with the labor correctly after endogonizing
    n1 = np.copy(nvec)
    n2 = np.zeros(S)
    n2[:-1] = nvec[1:]
    c, cmask = consumption(w, r, nvec, p, savings_guess, minimum)
    error1 = foc_k(r,c)
    #print cmask
    #print error1
    #error1[cmask[:-1]] = 10000
    #print error1
    #raw_input()
    return error1

def hh_ss_consumption(c, cum_p, prices):
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
    Returns the Capital Demand of the firms
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

def ss_solve_convex(rw_init,nvec):
    error = 1
    r_guess = rw_init[0]
    w_guess = rw_init[1]
    while error > .000001:
        prices = firm_price(r_guess,w_guess)
        minimum = min_consump(prices)
        com_price = comp_price(prices)
        guessvec = np.array((.2,.3))
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

def ss_solve_fsolve(rw_init,nvec):
    r_guess = rw_init[0]
    w_guess = rw_init[1]
    while error > .0001:
        prices = firm_price(r_guess,w_guess)
        minimum = min_consump(prices)
        com_price = comp_price(prices)
        guessvec = np.array((.2,.3))
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
        r_guess = r_new
        w_guess = w_new
    return np.array(([r_new, w_new]))
nvec = np.array((1.,1.,.2))
'''
w_guess = .4
r_guess = .1
guessvec = np.array(([.4,.1]))
prices = firm_price(r_guess,w_guess)
minimum = min_consump(prices)
com_price = comp_price(prices)
guessvec = np.array((.2,.3))
newb = opt.fsolve(savings_euler, guessvec, args = (r_guess, w_guess, com_price, minimum))
c_guess = consumption(w_guess, r_guess, nvec, com_price, newb, minimum)
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
print 'prices: {}, Consumption: {}, K: {}'.format(prices, Ybar, K_demand)
r_new = calc_new_r(prices[0], Ybar[0], K_demand[0])
w_new = calc_new_w(prices[0], Ybar[0], L_demand[0])
print 'w', w_new
print 'r', r_new
diff = np.array(([abs(r_guess-r_new),abs(w_guess-w_new)]))
print diff
'''

nvec = np.array((1.,1.,.2))
rw = np.array(([.4,.1]))
print ss_solve_convex(rw, nvec)
