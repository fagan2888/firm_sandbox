'''
------------------------------------------------------------------------
Last updated: 7/13/2015

Calculates steady state of OLG model with S age cohorts, J types, mortality risk

This py-file calls the following other file(s):

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy.optimize as opt
import pickle
#import income
#import demographics
import numpy.polynomial.polynomial as poly



'''
------------------------------------------------------------------------
Setting up the Model
------------------------------------------------------------------------
S            = number of periods an individual lives
J            = number of different ability groups
T            = number of time periods until steady state is reached
bin_weights  = percent of each age cohort in each ability group
starting_age = age of first members of cohort
ending age   = age of the last members of cohort
E            = number of cohorts before S=1
beta_annual  = discount factor for one year
beta         = discount factor for each age cohort
sigma        = coefficient of relative risk aversion
alpha        = capital share of income
nu_init      = contraction parameter in steady state iteration process
               representing the weight on the new distribution gamma_new
A            = total factor productivity parameter in firms' production
               function
delta_annual = depreciation rate of capital for one year
delta        = depreciation rate of capital for each cohort
ctilde       = minimum value amount of consumption
bqtilde      = minimum bequest value
ltilde       = measure of time each individual is endowed with each
               period
chi_n        = discount factor of labor
chi_b        = discount factor of incidental bequests
eta          = Frisch elasticity of labor supply
g_y_annual   = annual growth rate of technology
g_y          = growth rate of technology for one cohort
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPImindist   = Cut-off distance between iterations for TPI
------------------------------------------------------------------------
'''
#computational parameters
maxiter = 1000
mindist_SS = 1e-9
mu = 0.5

# Parameters
sigma = 2.0 # coeff of relative risk aversion for hh
beta = 0.98 # discount rate
delta = 0.1 # depreciation rate
alpha = 0.3 # capital's share of output
nu = 1.9 # elasticity of labor supply 
chi_n = 0.5 #utility weight, disutility of labor
chi_b = 0.2 #utility weight, warm glow bequest motive
ltilde = 1.0 # maximum hours
e = [0.5, 1.0, 1.2, 1.5] # effective labor units for the J types
#e = [1.0, 1.0, 1.0, 1.0] # effective labor units for the J types
S = 5 # periods in life of hh
J = 4 # number of lifetime income groups
surv_rate = np.array([0.99, 0.98, 0.6, 0.4, 0.0]) # probability of surviving to next period
#surv_rate = np.array([1.0, 1.0, 1.0, 1.0, 0.0]) # probability of surviving to next period
mort_rate = 1.0-surv_rate # probability of dying at the end of current period
surv_rate[-1] = 0.0
mort_rate[-1] = 1.0
surv_mat = np.tile(surv_rate.reshape(S,1),(1,J)) # matrix of survival rates
mort_mat = np.tile(mort_rate.reshape(S,1),(1,J)) # matrix of mortality rates
surv_rate1 = np.ones((S,1))# prob start at age S
surv_rate1[1:,0] = np.cumprod(surv_rate[:-1], dtype=float)
omega = np.ones((S,J))*surv_rate1# number of each age alive at any time
lambdas = np.array([0.5, 0.2, 0.2, 0.1])# fraction of each cohort of each type
weights = omega*lambdas/((omega*lambdas).sum()) # weights - dividing so weights sum to 1

# Functions and Definitions

print('checking omega')
omega2 = np.ones((S,1))# prob start at age S
omega2[1:,0] = np.cumprod(surv_rate[:-1], dtype=float)
print((omega[:,0].reshape(S,1)-omega2.reshape(S,1)).max())

def get_Y(K, L):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    Y = (K ** alpha) * (L ** (1 - alpha))
    return Y


def get_w(Y, L):
    '''
    Parameters: Aggregate output, Aggregate labor

    Returns:    Returns to labor
    '''
    w = (1 - alpha) * Y / L
    return w


def get_r(Y, K):
    '''
    Parameters: Aggregate output, Aggregate capital

    Returns:    Returns to capital
    '''
    r = (alpha * (Y / K)) - delta
    return r


def get_L(n):
    '''
    Parameters: n 

    Returns:    Aggregate labor
    '''
    L = np.sum(weights*(n*e))
    return L
    
def get_K(k):
    '''
    Parameters: k 

    Returns:    Aggregate capital
    '''
    K_constr = False
    K = np.sum(weights*k)
    if K <= 0:
        print 'b matrix and/or parameters resulted in K<=0'
        K_constr = True
    return K, K_constr



def MUc(c):
    '''
    Parameters: Consumption

    Returns:    Marginal Utility of Consumption
    '''
    output = c**(-sigma)
    return output


def MUl(n):
    '''
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    '''
    output =  -chi_n * ((ltilde-n) ** (-nu))
    return output

def MUb(bq):
    '''
    Parameters: Intentional bequests

    Returns:    Marginal Utility of Bequest
    '''
    output = chi_b * (bq ** (-sigma))
    return output
    
def get_BQ(r, k, j):
    '''
    Parameters: Distribution of capital stock (SxJ)

    Returns:    Bequests by ability (Jx1)
    '''

    output = (1 + r) * (k*weights[:,j].reshape(S,1)*mort_mat[:,j].reshape(S,1)).sum()

    return output
    
    
    
def get_dist_bq(BQ,j):
    '''
    Parameters: Aggregate bequests by ability type

    Returns:    Bequests by age and ability
    '''
    
    output = np.tile(BQ/(weights[:,j].sum(0)),(S,1))

    return output
    
def get_cons(w, r, n, k0, k, bq, j):
    '''
    Parameters: Aggregate bequests by ability type

    Returns:    Bequests by age and ability
    '''
    output = ((1+r)*k0) + w*n*e[j] - k + bq
    return output
    

def foc_k(r, c, j):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        L_guess  = distribution of labor (SxJ array)
        K_guess  = distribution of capital at the end of period t (S x J array)
        bq       = distribution of bequests (S x J array)

    Returns:
        Value of foc error ((S-1)xJ array)
    '''
    error = MUc(c[:-1,0]) - (1+r)*beta*surv_mat[:-1,j]*MUc(c[1:,0]) 
    return error


def foc_l(w, L_guess, c, j):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        L_guess  = distribution of labor (SxJ array)
        K_guess  = distribution of capital at the end of period t (S x J array)
        bq       = distribution of bequests (S x J array)

    Returns:
        Value of foc error (SxJ array)
    '''
    
    error = w*MUc(c)*e[j] + MUl(L_guess) 
    return error

def foc_bq(K_guess, c):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        L_guess  = distribution of labor (SxJ array)
        K_guess  = distribution of capital in period t (S-1 x J array)
        bq       = distribution of bequests (S x J array)

    Returns:
        Value of Euler error.
    '''
    error = MUc(c[-1,:]) -  MUb(K_guess[-1, :])
    return error


def perc_dif_func(simul, data):
    '''
    Used to calculate the absolute percent difference between the data
    moments and model moments
    '''
    frac = (simul - data)/data
    output = np.abs(frac)
    return output

def solve_hh(guesses, r, w, j):
    '''
    Parameters: SS interest rate (r), SS wage rate (w)
    Returns:    Savings (SxJ)
                Labor supply (SxJ)    

    '''
    
    k = guesses[0: S].reshape((S, 1))
    n = guesses[S:].reshape((S, 1))        
    BQ = get_BQ(r, k, j)
    bq = get_dist_bq(BQ,j)
    k0 = np.zeros((S,1))
    k0[1:,0] = k[:-1,0] # capital start period with
    c = get_cons(w, r, n, k0, k, bq, j)
    error1 = foc_k(r, c, j) 
    error2 = foc_l(w, n, c, j) 
    error3 = foc_bq(k, c) 

    # Check and punish constraing violations
    mask1 = n <= 0
    mask2 = n > ltilde
    mask4 = c <= 0
    mask3 = k < 0
    #mask3 = k[:-1,0] <= 0
    error2[mask1] += 1e14
    error2[mask2] += 1e14
    error1[mask3[:-1,0]] += 1e14
    error1[mask4[:-1,0]] += 1e14
    if k[-1,0] < 0:
        error3 += 1e14
    if c[-1,0] <= 0:
        error3 += 1e14

    #error3[mask3[-1,0]] += 1e14


    #print('max euler error')
    #print(max(list(error1.flatten()) + list(error2.flatten()) + list(error3.flatten())))
    return list(error1.flatten()) + list(error2.flatten()) + list(error3.flatten()) 


def Steady_State(guesses, mu):
    '''
    Parameters: Steady state distribution of capital guess as array
                size SxJ and labor supply array of SxJ rss
    Returns:    Array of SxJ * 2 Euler equation errors
    '''
    
    r = guesses[0]
    w = guesses[1]
    
    
    dist = 10
    iteration = 0
    dist_vec = np.zeros(maxiter)
    
    # Make initial guesses for capital and labor
    K_guess_init = np.ones((S, J)) * 0.05
    L_guess_init = np.ones((S, J)) * 0.3
    k = np.zeros((S,J)) # initialize k matrix
    n = np.zeros((S,J)) # initialize n matrix
    
    while (dist > mindist_SS) and (iteration < maxiter):
        

        
        for j in xrange(J):
            # Solve the euler equations
            if j == 0:
                guesses = np.append(K_guess_init[:,j], L_guess_init[:,j])
            else:
                guesses = np.append(k[:,(j-1)], n[:,(j-1)])
            solutions = opt.fsolve(solve_hh, guesses, args=(r, w, j), xtol=1e-9, col_deriv=1)
            #out = opt.fsolve(solve_hh, guesses, args=(r, w, j), xtol=1e-9, col_deriv=1, full_output=1)
            #print'solution found flag', out[2], out[3]
            #solutions = out[0]
            k[:,j] = solutions[:S].reshape(S)
            n[:,j] = solutions[S:].reshape(S)

        
        K, K_constr = get_K(k)
        L = get_L(n)
        Y = get_Y(K, L)    
        r_new = get_r(Y,K)
        w_new = get_w(Y,L)
        
        
        r = mu*r_new + (1-mu)*r # so if r low, get low save, so low capital stock, so high mpk, so r_new bigger
        w = mu*w_new + (1-mu)*w

        dist = np.array([perc_dif_func(r_new, r)]+[perc_dif_func(w_new, w)]).max()
        
        dist_vec[iteration] = dist
        if iteration > 10:
            if dist_vec[iteration] - dist_vec[iteration-1] > 0:
                mu /= 2.0
                print 'New value of mu:', mu
        iteration += 1
        print "Iteration: %02d" % iteration, " Distance: ", dist

 
    return [r, w]
    

# Make initial guesses for factor prices
r_guess_init = 0.4
w_guess_init = 1.0
guesses = [r_guess_init, w_guess_init]

# Solve SS
solutions = Steady_State(guesses, mu)

rss = solutions[0]
wss = solutions[1]
print 'rss, wss: ', rss, wss

K_guess_init = np.ones((S, J)) * 0.05
L_guess_init = np.ones((S, J)) * 0.3
Kssmat = np.zeros((S, J))
Lssmat = np.zeros((S, J))
Cssmat = np.zeros((S, J))
error1 = np.zeros((S-1,J)) # initialize foc k errors
error2 = np.zeros((S,J)) # initialize foc k errors
error3 = np.zeros((1,J)) # initialize foc k errors

for j in xrange(J):
    if j == 0:
        guesses = np.append(K_guess_init[:,j], L_guess_init[:,j])
    else:
        guesses = np.append(Kssmat[:,(j-1)], Lssmat[:,(j-1)])
    #solutions = opt.fsolve(solve_hh, guesses, args=(rss, wss, j), xtol=1e-9, col_deriv=1)
    out = opt.fsolve(solve_hh, guesses, args=(rss, wss, j), xtol=1e-9, col_deriv=1, full_output=1)
   # print'solution found flag', out[2], out[3]
    #print 'fsovle output: ', out[1]
    solutions = out[0]
    Kssmat[:,j] = solutions[:S].reshape(S)
    Lssmat[:,j] = solutions[S:].reshape(S)
    BQss = get_BQ(rss, Kssmat[:,j].reshape(S,1), j)
    bqss = get_dist_bq(BQss, j).reshape(S,1)
    k0ss = np.zeros((S,1))
    k0ss[1:,0] = Kssmat[:-1,j] # capital start period with
    Cssmat[:,j] = get_cons(wss, rss, Lssmat[:,j].reshape(S,1), k0ss[:,0].reshape(S,1), Kssmat[:,j].reshape(S,1), bqss, j).reshape(S)
    # check Euler errors
    error1[:,j] = foc_k(rss, Cssmat[:,j].reshape(S,1), j).reshape(S-1) 
    error2[:,j] = foc_l(wss, Lssmat[:,j].reshape(S,1), Cssmat[:,j].reshape(S,1), j).reshape(S) 
    error3[:,j] = foc_bq(Kssmat[:,j].reshape(S,1), Cssmat[:,j].reshape(S,1))


Lss = get_L(Lssmat)  
Kss, K_constr = get_K(Kssmat) 
Yss = get_Y(Kss, Lss) 

Css = np.sum(weights*Cssmat)

print 'RESOURCE CONSTRAINT DIFFERENCE:', Yss - Css- delta*Kss

# check Euler errors
print("Euler errors")
print(error1)
print(error2)
print(error3)

print 'Kssmat: ', Kssmat