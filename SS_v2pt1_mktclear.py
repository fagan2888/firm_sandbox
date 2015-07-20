'''
------------------------------------------------------------------------
Last updated: 7/13/2015

Calculates steadX state of OLG model with S age cohorts, J tXpes, mortalitX risk

This pX-file calls the following other file(s):

This pX-file creates the following other file(s):
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
eta          = Frisch elasticitX of labor supply
g_y_annual   = annual growth rate of technology
g_y          = growth rate of technologX for one cohort
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPImindist   = Cut-off distance between iterations for TPI
------------------------------------------------------------------------
'''


# Parameters
sigma = 1.9 # coeff of relative risk aversion for hh
beta = 0.98 # discount rate
alpha = 1.0 # preference parameter - share of good 1 in composite consumption
cbar1 = 0.0 # min cons of good 1
cbar2 = 0.0 #min cons of good 2
delta = 0.1 # depreciation rate
A = 1.0 # Total factor productivity
gamma = 0.3 # capital's share of output
#xi = np.array([[0.2, 0.8],[0.3, 0.7]]) # fixed coeff input-output matrix
xi = np.array([[0.0, 1.0],[0.0, 1.0]]) # fixed coeff input-output matrix
epsilon = 0.6 # elasticity of substitution between capital and labor
nu = 2.0 # elasticity of labor supply 
chi_n = 0.5 #utility weight, disutility of labor
chi_b = 0.2 #utility weight, warm glow bequest motive
ltilde = 1.0 # maximum hours
#e = [0.5, 1.0, 1.2, 1.5] # effective labor units for the J types
e = [1.0, 1.0, 1.0, 1.0] # effective labor units for the J types
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

def get_X(K, L):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    #X = (K ** alpha) * (L ** (1 - alpha))
    X = (A * (((gamma**(1/epsilon))*(K**((epsilon-1)/epsilon))) + 
          (((1-gamma)**(1/epsilon))*(L**((epsilon-1)/epsilon))))**(epsilon/(epsilon-1)))
    return X


def get_w(X, L):
    '''
    Parameters: Aggregate output, Aggregate labor

    Returns:    Returns to labor
    '''
    #w = (1 - alpha) * X / L
    w = (A**((epsilon-1)/epsilon))*((((1-gamma)*X)/L)**(1/epsilon)) 
    return w


def get_r(X, K):
    '''
    Parameters: Aggregate output, Aggregate capital

    Returns:    Returns to capital
    '''
    #r = (alpha * (X / K)) - delta
    r = (A**((epsilon-1)/epsilon))*(((gamma*X)/K)**(1/epsilon)) - delta
    return r


def get_L(n):
    '''
    Parameters: n 

    Returns:    Aggregate labor
    '''
    L = np.sum(weights*n*e)
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

def get_C(c):
    '''
    Parameters: c 

    Returns:    Aggregate consumption
    '''
    C = np.sum(weights*c)

    return C

def get_p_c(r, w):
    '''
    Generates price of consumption good/producer output

    Returns: p_c
    '''
    p_c = (A**(epsilon-1))*(((1-gamma)*(w**(1-epsilon))) + (gamma*((r+delta)**(1-epsilon))))
    return p_c
    
def get_p_tilde(p_c1, p_c2):
    #p_tilde = ((p_c1/alpha)**alpha)*((p_c2/(1-alpha))**(1-alpha))
    p_tilde = ((p_c1/alpha)**alpha)
    return p_tilde

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
    
def get_BQ(r, k):
    '''
    Parameters: Distribution of capital stock (SxJ)

    Returns:    Bequests by ability (Jx1)
    '''
    output = (1 + r) * (k*weights*mort_mat).sum(0)

    return output
    
    
    
def get_dist_bq(BQ):
    '''
    Parameters: Aggregate bequests by ability type

    Returns:    Bequests by age and ability
    '''
    output = np.tile(BQ.reshape(1, J)/weights.sum(0),(S,1))

    return output
    
def get_cons(w, r, n, k0, k, bq, p_c1, p_c2, p_tilde):
    '''
    Parameters: Aggregate bequests by ability type

    Returns:    Bequests by age and ability
    '''
    output = (((1+r)*k0) + w*n*e - k + bq - (p_c1*cbar1) - (p_c2*cbar2))/p_tilde

    return output
    

def get_k_demand(r,X):
    '''
    Parameters: Interest rate
                Output

    Returns:    Demand for capital by the firm
    '''
    output = (gamma*X)/(((r+delta)**epsilon)*(A**(1-epsilon)))

    return output

def foc_k(r, c):
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

    error = MUc(c[:-1,:]) - (1+r)*beta*surv_mat[:-1,:]*MUc(c[1:,:]) 
    return error


def foc_l(w, L_guess, c):
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
    
    error = w*MUc(c)*e + MUl(L_guess) 
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

def solve_hh(guesses, r, w, p_c1, p_c2, p_tilde):
    '''
    Parameters: SS interest rate (r), SS wage rate (w)
    Returns:    Savings (SxJ)
                Labor supply (SxJ)    

    '''
    k = guesses[0: S * J].reshape((S, J))
    n = guesses[S * J:].reshape((S, J))
    BQ = get_BQ(r, k)
    bq = get_dist_bq(BQ)
    k0 = np.zeros((S,J))
    k0[1:,:] = k[:-1,:] # capital start period with
    c = get_cons(w, r, n, k0, k, bq, p_c1, p_c2, p_tilde)
    error1 = foc_k(r, c) 
    error2 = foc_l(w, n, c) 
    error3 = foc_bq(k, c)  

    # Check and punish constraing violations
    mask1 = n < 0
    error2[mask1] += 1e9
    mask2 = n > ltilde
    error2[mask2] += 1e9
    if k.sum() <= 0:
        error1 += 1e9
    mask3 = c < 0
    error2[mask3] += 1e9
    # mask4 = np.diff(L_guess) > 0
    # error2[mask4] += 1e9
    #return list(k.flatten()) + list(n.flatten()) 
    #print'max euler error', np.array(list(error1.flatten()) + list(error2.flatten())).max()
    return list(error1.flatten()) + list(error2.flatten()) + list(error3.flatten()) 

def solve_output(guesses, r, X_c_1, X_c_2):
    X_1 = guesses[0]
    X_2 = guesses[1]
    I1 = delta*get_k_demand(r,X_1) # investment demand - will differ not in SS
    I2 = delta*get_k_demand(r,X_2) #investment demand 
    error1 = X_c_1  + (I1*xi[0,0]) + (I2*xi[1,0]) - X_1
    error2 = X_c_2  + (I1*xi[0,1]) + (I2*xi[1,1]) - X_2  

    #print 'solve_ouput errors: ', error1, error2
    return [error1, error2]

def Steady_State(guesses):
    '''
    Parameters: Steady state distribution of capital guess as array
                size SxJ and labor supply array of SxJ rss
    Returns:    Array of SxJ * 2 Euler equation errors
    '''
    
    r = guesses[0]
    w = guesses[1]

    # find prices of consumption goods
    p_c1 = get_p_c(r,w)
    p_c2 = get_p_c(r,w)
    p_tilde = get_p_tilde(p_c1,p_c2)

    # Make initial guesses for capital and labor
    K_guess_init = np.ones((S, J)) * 0.5
    L_guess_init = np.ones((S, J)) * 0.3
    guesses = np.append(K_guess_init, L_guess_init)
    solutions = opt.fsolve(solve_hh, guesses, args=(r, w, p_c1, p_c2, p_tilde), xtol=1e-9, col_deriv=1)
    #solutions = solve_hh(guesses,r, w, p_c1, p_c2, p_tilde)
    #out = opt.fsolve(solve_hh, guesses, args=(r, w, p_c1, p_c2, p_tilde), xtol=1e-9, col_deriv=1, full_output=1)
    #print'solution found flag', out[2], out[3]
    #solutions = out[0]
    #k = solutions[0:S * J].reshape(S, J)
    #n = solutions[S * J:].reshape(S, J)
    k = np.array(solutions[0:S * J]).reshape(S,J)
    n = np.array(solutions[S * J:]).reshape(S,J)


    # Find consumption from HH in SS
    BQ = get_BQ(r, k)
    bq = get_dist_bq(BQ)
    k0 = np.zeros((S,J))
    k0[1:,:] = k[:-1,:] # capital start period with
    c = get_cons(w, r, n, k0, k, bq, p_c1, p_c2, p_tilde)
    c1 = (p_tilde*c*alpha)/p_c1 + cbar1
    c2 = (p_tilde*c*(1-alpha))/p_c2 + cbar2


    # Find total consumption of each good
    C1 = get_C(c1)
    C2 = get_C(c2)

    # Find total demand for output from each sector from consumption
    X_c_1 = C1
    X_c_2 = C2

    guesses = [(X_c_1+X_c_2)/2, (X_c_1+X_c_2)/2]
    x_sol = opt.fsolve(solve_output, guesses, args=(r, X_c_1, X_c_2), xtol=1e-9, col_deriv=1)

    X1 = x_sol[0]
    X2 = x_sol[1]

    K, K_constr = get_K(k)
    L = get_L(n)

    # Find demand for labor and capital from each industry
    K1 = (gamma*X1)/(((r+delta)**(epsilon))*(A**(1-epsilon)))
    K2 = (gamma*X2)/(((r+delta)**(epsilon))*(A**(1-epsilon)))
    L1 = ((1-gamma)*X1)/(w*(A**(1-epsilon)))
    L2 = ((1-gamma)*X2)/(w*(A**(1-epsilon)))

    # Check market clearing   
    error1 = K-K1-K2
    error2 = L-L1-L2


    # Check and punish constraing violations
    if r <= 0:
        error1 += 1e9
    if r > 1:
        error1 += 1e9
    if w <= 0:
        error2 += 1e9
    #print('errors')
    #print(error1)
    #print(error2)
    return [r, w]
    #return [error1, error2]
    

# Make initial guesses for factor prices
r_guess_init = 0.4
w_guess_init = 1.0
guesses = [r_guess_init, w_guess_init]

# Solve SS
solutions = opt.fsolve(Steady_State, guesses, xtol=1e-9, col_deriv=1)
#solutions = Steady_State(guesses)
rss = solutions[0]
wss = solutions[1]
print 'ss r, w: ', rss, wss

p_c1_ss = get_p_c(rss,wss)
p_c2_ss = get_p_c(rss,wss)
p_tilde_ss = get_p_tilde(p_c1_ss,p_c2_ss)
print 'SS cons prices: ', p_c1_ss, p_c2_ss, p_tilde_ss

K_guess_init = np.ones((S, J)) * 0.05
L_guess_init = np.ones((S, J)) * 0.3
guesses = np.append(K_guess_init, L_guess_init)
ss_vars = opt.fsolve(solve_hh, guesses, args=(rss, wss, p_c1_ss, p_c2_ss, p_tilde_ss), xtol=1e-9, col_deriv=1)
Kssmat = ss_vars[0:S * J].reshape(S, J)
Lssmat = ss_vars[S * J:].reshape(S, J)
Lss = get_L(Lssmat)  
Kss, K_constr = get_K(Kssmat) 

# Find consumption from HH in SS
BQss = get_BQ(rss, Kssmat)
bq_ss = get_dist_bq(BQss)
k0_ss = np.zeros((S,J))
k0_ss[1:,:] = Kssmat[:-1,:] # capital start period with
css = get_cons(wss, rss, Lssmat, k0_ss, Kssmat, bq_ss, p_c1_ss, p_c2_ss, p_tilde_ss)
c1ss = (p_tilde_ss*css*alpha)/p_c1_ss + cbar1
c2ss = (p_tilde_ss*css*(1-alpha))/p_c2_ss + cbar2

# Find total consumption of each good
C1ss = get_C(c1ss)
C2ss = get_C(c2ss)

# Find total demand for output from each sector from consumption
X_c_1_ss = C1ss
X_c_2_ss = C2ss

print 'X_c_2_ss', X_c_2_ss

guesses = [(X_c_1_ss+X_c_2_ss)/2, (X_c_1_ss+X_c_2_ss)/2]
x_sol_ss = opt.fsolve(solve_output, guesses, args=(rss, X_c_1_ss, X_c_2_ss), xtol=1e-9, col_deriv=1)

X_1_ss = x_sol_ss[0]
X_2_ss = x_sol_ss[1]

#### Need to solve for K1, L1 here
K1ss = (X_1_ss/(X_1_ss+X_2_ss))*Kss
L1ss = (X_1_ss/(X_1_ss+X_2_ss))*Lss
K2ss = Kss -K1ss
L2ss = Lss-L1ss

Y1ss = get_X(K1ss,L1ss)
Y2ss = get_X(K2ss,L2ss)

print 'cons: ', C1ss, C2ss
print 'Kss: ', K1ss, K2ss
print 'Lss: ', L1ss, L2ss
#print 'K/L: ', K1ss/L1ss, K2ss/L2ss
print 'Xss: ', X_1_ss, X_2_ss, Y1ss, Y2ss
print 'test x: ', X_2_ss, delta*(K1ss+K2ss)
print 'test k: ', X_2_ss, (delta*K1ss)/(1-((gamma*delta)/((rss+delta)**epsilon)))

I1ss = delta*get_k_demand(rss,X_1_ss) # investment demand - will differ not in SS
I2ss = delta*get_k_demand(rss,X_2_ss)

X1ss_check = X_c_1_ss  + (I1ss*xi[0,0]) + (I2ss*xi[1,0])
X2ss_check = X_c_2_ss  + (I1ss*xi[0,1]) + (I2ss*xi[1,1])
print 'X1 check: ', X_1_ss, X1ss_check
print 'X2 check: ', X_2_ss, X2ss_check


print 'RESOURCE CONSTRAINT DIFFERENCE:'
print 'RC1: ', X_1_ss - Y1ss
print 'RC2: ', X_2_ss - Y2ss
print 'RC1: ', X_1_ss - C1ss- delta*K1ss*xi[0,0] - delta*K2ss*xi[1,0]
print 'RC2: ', X_2_ss - C2ss- delta*K1ss*xi[0,1] - delta*K2ss*xi[1,1]

# check Euler errors
error1 = foc_k(rss, css) 
error2 = foc_l(wss, Lssmat, css) 
error3 = foc_bq(Kssmat, css) 

print("Euler errors")
print(error1)
print(error2)
print(error3)

print 'kssmat: ', Kssmat


