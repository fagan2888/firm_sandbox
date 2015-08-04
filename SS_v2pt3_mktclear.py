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
alpha = 0.29 # preference parameter - share of good 1 in composite consumption
cbar1 = 0.000 # min cons of good 1
cbar2 = 0.000 #min cons of good 2
delta = 0.1 # depreciation rate
A = 1.0 # Total factor productivity
gamma = 0.3 # capital's share of output
xi = np.array([[0.2, 0.8],[0.3, 0.7]]) # fixed coeff input-output matrix
pi = np.array([[0.5, 0.5],[0.1, 0.9]]) # fixed coeff pce-bridge matrix relating output and cons goods
#xi = np.array([[1.0, 0.0],[0.0, 1.0]]) # fixed coeff input-output matrix
#xi = np.array([[0.0, 1.0],[0.0, 1.0]]) # fixed coeff input-output matrix
epsilon = 0.6 # elasticity of substitution between capital and labor
nu = 2.0 # elasticity of labor supply 
chi_n = 0.5 #utility weight, disutility of labor
chi_b = 0.2 #utility weight, warm glow bequest motive
ltilde = 1.0 # maximum hours
e = [0.5, 1.0, 1.2, 1.7] # effective labor units for the J types
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

def get_X(K, L):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    #X = (K ** alpha) * (L ** (1 - alpha))
    X = (A * (((gamma**(1/epsilon))*(K**((epsilon-1)/epsilon))) + 
          (((1-gamma)**(1/epsilon))*(L**((epsilon-1)/epsilon))))**(epsilon/(epsilon-1)))
    return X


def get_w(X, L, p_c):
    '''
    Parameters: Aggregate output, Aggregate labor

    Returns:    Returns to labor
    '''
    #w = (1 - alpha) * X / L
    w = p_c*((A**((epsilon-1)/epsilon))*((((1-gamma)*X)/L)**(1/epsilon))) 
    return w


def get_r(X, K, p_c):
    '''
    Parameters: Aggregate output, Aggregate capital

    Returns:    Returns to capital
    '''
    #r = (alpha * (X / K)) - delta
    r = p_c*((A**((epsilon-1)/epsilon))*(((gamma*X)/K)**(1/epsilon))) - delta
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
    p_c = (((1-gamma)*((w/A)**(1-epsilon)))+(gamma*(((r+delta)/A)**(1-epsilon))))**(1/(1-epsilon))
    return p_c
    
def get_p_tilde(p_c1, p_c2):
    p_tilde = ((p_c1/alpha)**alpha)*((p_c2/(1-alpha))**(1-alpha))
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
    
def get_BQ(r, k, j):
    '''
    Parameters: Distribution of capital stock (SxJ)

    Returns:    Bequests by ability (Jx1)
    '''

    output = (1 + r) * (k*weights[:,j].reshape(S,1)*mort_mat[:,j].reshape(S,1)).sum()

    return output
    
def get_dist_bq(BQ, j):
    '''
    Parameters: Aggregate bequests by ability type

    Returns:    Bequests by age and ability
    '''
    output = np.tile(BQ/(weights[:,j].sum(0)),(S,1))

    return output

def get_cons(w, r, n, k0, k, bq, p_c1, p_c2, p_tilde, j):
    '''
    Parameters: Aggregate bequests by ability type

    Returns:    Bequests by age and ability
    '''

    output = (((1+r)*k0) + w*n*e[j] - k + bq - (p_c1*cbar1) - (p_c2*cbar2))/p_tilde

    return output
    

def get_k_demand(w,r,X):
    '''
    Parameters: Interest rate
                Output

    Returns:    Demand for capital by the firm
    '''
    #output = (gamma*X)/(((r+delta)**epsilon)*(A**(1-epsilon)))
    output = (X*(A**((1-epsilon)/epsilon)))/(((gamma**(1/epsilon))+
              (((1-gamma)**(1/epsilon))*(((r+delta)/w)**(epsilon-1))*
              (((1-gamma)/gamma)**((epsilon-1)/epsilon))))**(epsilon/(epsilon-1)))

    return output

def get_l_demand(w,r,K):
    '''
    Parameters: Wage rate
                Capital demand

    Returns:    Demand for labor by the firm
    '''
    output = K*((1-gamma)/gamma)*(((r+delta)/w)**epsilon)

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


def foc_l(w, L_guess, c, p_tilde, j):
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
    
    error = (w*MUc(c)*e[j])/p_tilde + MUl(L_guess) 
    return error

def foc_bq(K_guess, c, p_tilde):
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
    error = MUc(c[-1,:])/p_tilde -  MUb(K_guess[-1, :])
    return error


def solve_hh(guesses, r, w, p_c1, p_c2, p_tilde, j):
    '''
    Parameters: SS interest rate (r), SS wage rate (w)
    Returns:    Savings (Sx1)
                Labor supply (Sx1)    

    '''
    k = guesses[0: S].reshape((S, 1))
    n = guesses[S:].reshape((S, 1))        
    BQ = get_BQ(r, k, j)
    bq = get_dist_bq(BQ,j)
    k0 = np.zeros((S,1))
    k0[1:,0] = k[:-1,0] # capital start period with
    c = get_cons(w, r, n, k0, k, bq, p_c1, p_c2, p_tilde, j)
    error1 = foc_k(r, c, j) 
    error2 = foc_l(w, n, c, p_tilde, j) 
    error3 = foc_bq(k, c, p_tilde) 

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


def solve_output(guesses, w, r, X_c_1, X_c_2):
    X_1 = guesses[0]
    X_2 = guesses[1]
    I1 = delta*get_k_demand(w, r,X_1) # investment demand - will differ not in SS
    I2 = delta*get_k_demand(w, r,X_2) #investment demand 
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
    K_guess_init = np.ones((S, J)) * 0.05
    L_guess_init = np.ones((S, J)) * 0.3
    #guesses = list(K_guess_init.flatten()) + list(L_guess_init.flatten())

    # solve hh problem for consumption, labor supply, and savings
    k = np.zeros((S, J))
    n = np.zeros((S, J))
    c = np.zeros((S, J))
    for j in xrange(J):
        if j == 0:
            guesses = np.append(K_guess_init[:,j], L_guess_init[:,j])
        else:
            guesses = np.append(k[:,(j-1)], n[:,(j-1)])
        solutions = opt.fsolve(solve_hh, guesses, args=(r, w, p_c1, p_c2, p_tilde, j), xtol=1e-9, col_deriv=1)
        #out = opt.fsolve(solve_hh, guesses, args=(r, w, j), xtol=1e-9, col_deriv=1, full_output=1)
        #print'solution found flag', out[2], out[3]
        #solutions = out[0]
        k[:,j] = solutions[:S].reshape(S)
        n[:,j] = solutions[S:].reshape(S)
        BQ = get_BQ(r, k[:,j].reshape(S,1), j)
        bq = get_dist_bq(BQ, j).reshape(S,1)
        k0 = np.zeros((S,1))
        k0[1:,0] = k[:-1,j] # capital start period with
        c[:,j] = get_cons(w, r, n[:,j].reshape(S,1), k0[:,0].reshape(S,1), k[:,j].reshape(S,1), bq, p_c1, p_c2, p_tilde, j).reshape(S)

    c1 = (p_tilde*c*alpha)/p_c1 + cbar1
    c2 = (p_tilde*c*(1-alpha))/p_c2 + cbar2

    # Find total consumption of each good
    C1 = get_C(c1)
    C2 = get_C(c2)

    # Find total demand for output from each sector from consumption
    X_c_1 = pi[0,0]*C1 + pi[1,0]*C2
    X_c_2 = pi[0,1]*C1 + pi[1,1]*C2

    guesses = [(X_c_1+X_c_2)/2, (X_c_1+X_c_2)/2]
    x_sol = opt.fsolve(solve_output, guesses, args=(w, r, X_c_1, X_c_2), xtol=1e-9, col_deriv=1)

    X1 = x_sol[0]
    X2 = x_sol[1]

    # find aggregate savings and labor supply
    K_s, K_constr = get_K(k)
    L_s = get_L(n)


    #### Need to solve for labor and capital demand from each industry
    K1_d = get_k_demand(w, r, X1)
    L1_d = get_l_demand(w, r, K1_d)
    K2_d = get_k_demand(w, r, X2)
    L2_d = get_l_demand(w, r, K2_d)

    #print 'r diffs', r-get_r(X1,K1_d, p_c1), r-get_r(X2,K2_d, p_c1)

    # Check labor and capital market clearing conditions
    K_d = K1_d + K2_d 
    L_d = L1_d + L2_d 
    
    error1 = K_s - K_d
    error2 = L_s - L_d

    # Check and punish violations
    if r <= 0:
        error1 += 1e9
    if r > 1:
        error1 += 1e9
    if w <= 0:
        error2 += 1e9

    return [error1, error2]
    

# Solve SS
r_guess_init = 0.77
w_guess_init = 1.03 
guesses = [r_guess_init, w_guess_init]
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
kss = np.zeros((S, J))
nss = np.zeros((S, J))
css = np.zeros((S, J))
error1 = np.zeros((S-1,J)) # initialize foc k errors
error2 = np.zeros((S,J)) # initialize foc k errors
error3 = np.zeros((1,J)) # initialize foc k errors

for j in xrange(J):
    if j == 0:
        guesses = np.append(K_guess_init[:,j], L_guess_init[:,j])
    else:
        guesses = np.append(kss[:,(j-1)], nss[:,(j-1)])
    #solutions = opt.fsolve(solve_hh, guesses, args=(rss, wss, j), xtol=1e-9, col_deriv=1)
    out = opt.fsolve(solve_hh, guesses, args=(rss, wss, p_c1_ss, p_c2_ss, p_tilde_ss, j), xtol=1e-9, col_deriv=1, full_output=1)
   # print'solution found flag', out[2], out[3]
    #print 'fsovle output: ', out[1]
    solutions = out[0]
    kss[:,j] = solutions[:S].reshape(S)
    nss[:,j] = solutions[S:].reshape(S)
    BQss = get_BQ(rss, kss[:,j].reshape(S,1), j)
    bqss = get_dist_bq(BQss, j).reshape(S,1)
    k0ss = np.zeros((S,1))
    k0ss[1:,0] = kss[:-1,j] # capital start period with
    css[:,j] = get_cons(wss, rss, nss[:,j].reshape(S,1), k0ss[:,0].reshape(S,1), kss[:,j].reshape(S,1), bqss, p_c1_ss, p_c2_ss, p_tilde_ss, j).reshape(S)
    # check Euler errors
    error1[:,j] = foc_k(rss, css[:,j].reshape(S,1), j).reshape(S-1) 
    error2[:,j] = foc_l(wss, nss[:,j].reshape(S,1), css[:,j].reshape(S,1), p_tilde_ss, j).reshape(S) 
    error3[:,j] = foc_bq(kss[:,j].reshape(S,1), css[:,j].reshape(S,1), p_tilde_ss)

c1ss = (p_tilde_ss*css*alpha)/p_c1_ss + cbar1
c2ss = (p_tilde_ss*css*(1-alpha))/p_c2_ss + cbar2

# Find total consumption of each good
C1ss = get_C(c1ss)
C2ss = get_C(c2ss)

# Find total demand for output from each sector from consumption
X_c_1_ss = pi[0,0]*C1ss + pi[1,0]*C2ss
X_c_2_ss = pi[0,1]*C1ss + pi[1,1]*C2ss

print 'X_c_2_ss', X_c_2_ss

guesses = [(X_c_1_ss+X_c_2_ss)/2, (X_c_1_ss+X_c_2_ss)/2]
x_sol_ss = opt.fsolve(solve_output, guesses, args=(wss, rss, X_c_1_ss, X_c_2_ss), xtol=1e-9, col_deriv=1)

X1_ss = x_sol_ss[0]
X2_ss = x_sol_ss[1]

# find aggregate savings and labor supply
K_s_ss, K_constr = get_K(kss)
L_s_ss = get_L(nss)

#### Need to solve for labor and capital demand from each industry
K1_d_ss = get_k_demand(wss, rss, X1_ss)
L1_d_ss = get_l_demand(wss, rss, K1_d_ss)
K2_d_ss = get_k_demand(wss, rss, X2_ss)
L2_d_ss = get_l_demand(wss, rss, K2_d_ss)

# Check labor and capital market clearing conditions
K_d_ss = K1_d_ss + K2_d_ss 
L_d_ss = L1_d_ss + L2_d_ss 

cap_diff = K_s_ss - K_d_ss
labor_diff = L_s_ss - L_d_ss
print 'Market clearing diffs: ', cap_diff, labor_diff

Y1ss = get_X(K1_d_ss,L1_d_ss)
Y2ss = get_X(K2_d_ss,L2_d_ss)

#print 'cons: ', C1ss, C2ss
#print 'Kss: ', K1_d_ss, K2_d_ss
#print 'Lss: ', L1_d_ss, L2_d_ss
#print 'K/L: ', K1_d_ss/L1_d_ss, K2_d_ss/L2_d_ss
#print 'Xss: ', X1_ss, X2_ss, Y1ss, Y2ss


I1ss = delta*get_k_demand(wss,rss,X1_ss) # investment demand - will differ not in SS
I2ss = delta*get_k_demand(wss,rss,X2_ss)

#X1ss_check = X_c_1_ss  + (I1ss*xi[0,0]) + (I2ss*xi[1,0])
#X2ss_check = X_c_2_ss  + (I1ss*xi[0,1]) + (I2ss*xi[1,1])
#print 'X1 check: ', X1_ss, X1ss_check
#print 'X2 check: ', X2_ss, X2ss_check


print 'RESOURCE CONSTRAINT DIFFERENCE:'
print 'RC1: ', X1_ss - Y1ss
print 'RC2: ', X2_ss - Y2ss
print 'RC1: ', X1_ss - X_c_1_ss- delta*K1_d_ss*xi[0,0] - delta*K2_d_ss*xi[1,0]
print 'RC2: ', X2_ss - X_c_2_ss - delta*K1_d_ss*xi[0,1] - delta*K2_d_ss*xi[1,1]


print("Euler errors")
print(error1)
print(error2)
print(error3)

print 'kssmat: ', kss


