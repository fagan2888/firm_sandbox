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
delta_annual = depreciation rate of capital for one year
delta        = depreciation rate of capital for each cohort
ctilde       = minimum value amount of consumption
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
alpha = 0.5 # preference parameter - share of good i in composite consumption, shape =(I,), shares must sum to 1
cbar = np.array([0.001, 0.001]) # min cons of each of I goods, shape =(I,)
delta = .05
epsilon = np.array((.55, .45))
gamma = np.array((.6, .5))
A = 1.0 # Total factor productivity
S = 3 # periods in life of hh
I = 2 # number of consumption goods
M = 2 # number of production industries

w_guess = .4
r_guess = .1
nvec = np.array((1.,1.,.2))

def cap_clear(L_demand, nvec):
    '''
    Checks the Capital market to see if it is cleared
    Eq. 11.36 in Rick's write-up
    Params
    L_demand - Labor demand for firms
    nvec - Labor supplied vector
    '''
    return np.sum(L_demand) - np.sum(nvec)

def lab_clear(K_demand, bvec):
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
    return ((price_vec[0]/alpha)**alpha)*(price_vec[1]/(1-alpha))**(1-alpha)

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
    return c 

def foc_k(r,c):
    '''
    Returns the first order condition vector for capital
    Params
    w - wage guess
    c - consumption vector
    '''
    error = c[:-1]**-sigma-(1+r)*beta*(c[1:])**-sigma
    print error 
    return error

def consump_euler(consump_guess, r, w, n, p, minimum, b):
    '''
    This willl solve for a new consumption vector of S length
    Eq. 11.45 in Rick's write up
    '''
    b1 = np.zeros(S)
    b2 = np.zeros(S)
    b3 = np.zeros(S)
    b1[1:]  = b
    b2[:-1] = b
    b3[:-2] = b[1:]
    print b1, b2
    
    return p*consump_guess + minimum + b2 - (1+r)*b1 + w*n
    

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
    c = consumption(w_guess, r_guess, nvec, p, savings_guess, minimum)
    #print c
    error1 = foc_k(r,c)
    #mask1 =  <= 0
    #print mask1
    #error1[mask1] += 1e10
    return error1


    
prices = firm_price(r_guess,w_guess)
#print prices
minimum = min_consump(prices)
com_price = comp_price(prices)
print com_price
guessvec = np.array((.2,.3))

newb = opt.fsolve(savings_euler, guessvec, args = (r_guess, w_guess, com_price, minimum))
#print newb

c_guess = consumption(w_guess, r_guess, nvec, com_price, newb, minimum)
newc = consump_euler(c_guess, r_guess, w_guess, nvec, com_price, minimum, newb)
#print (c_guess, r_guess, w_guess, nvec, com_price, minimum, newb)
#print newc



