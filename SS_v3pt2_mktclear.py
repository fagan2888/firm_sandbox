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
alpha = np.array([0.29, 0.2, (1-0.2-0.29)]) # preference parameter - share of good i in composite consumption, shape =(I,), shares must sum to 1
#alpha = 0.29 # preference parameter - share of good 1 in composite consumption
cbar = np.array([0.001, 0.002, 0.000]) # min cons of each of I goods, shape =(I,)
delta = np.array([0.1, 0.12, 0.15, 0.11]) # depreciation rate, shape =(M,)
#delta = np.array([0.1, 0.1]) # depreciation rate, shape =(M,)
#delta = 0.1 # depreciation rate
A = 1.0 # Total factor productivity
gamma = np.array([0.3, 0.25, 0.4, 0.33]) # capital's share of output, shape =(M,)
#gamma = np.array([0.3, 0.3])
#gamma = 0.3 # capital's share of output
xi = np.array([[0.2, 0.5, 0.2, 0.1],[0.0, 0.2, 0.8, 0.0], [0.4, 0.2, 0.2, 0.2], [0.3, 0.3, 0.1, 0.3] ]) # fixed coeff input-output matrix, shape =(M,M)
#xi = np.array([[0.2, 0.8],[0.3, 0.7]]) 
pi = np.array([[0.2, 0.3, 0.3, 0.2],[0.1, 0.8, 0.1, 0.0],[0.25, 0.25, 0.25, 0.25]]) # fixed coeff pce-bridge matrix relating output and cons goods, shape =(I,M)
#pi = np.array([[1.0, 0.0],[0.0, 1.0]]) # fixed coeff pce-bridge matrix relating output and cons goods, shape =(I,M)
epsilon = np.array([0.55, 0.6, 0.62, 0.6]) # elasticity of substitution between capital and labor, shape =(M,)
#epsilon = np.array([0.6, 0.6])
#epsilon = 0.6 # elasticity of substitution between capital and labor
nu = 2.0 # elasticity of labor supply 
chi_n = 0.5 #utility weight, disutility of labor
chi_b = 0.2 #utility weight, warm glow bequest motive
ltilde = 1.0 # maximum hours
e = [0.5, 1.0, 1.2, 1.7] # effective labor units for the J types
#e = [1.0, 1.0, 1.0, 1.0] # effective labor units for the J types
S = 5 # periods in life of hh
J = 4 # number of lifetime income groups
I = 3 # number of consumption goods
M = 4 # number of production industries
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


# Tax parameters
tau_b = np.ones(M)*0.0 #0.25
tau_d = np.ones(M)*0.0 #0.15 # without adjustment costs, want div tax rate to exceed cap gains rate
tau_g = np.ones(M)*0.0 #0.12 
delta_tau = delta#*1.2 # for not just make tax depreciation rate some scaled version of the rate of physical depreciation

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


def get_w(X, L, p):
    '''
    Parameters: Aggregate output, Aggregate labor

    Returns:    Returns to labor
    '''
    #w = (1 - alpha) * X / L
    w = p*((A**((epsilon-1)/epsilon))*((((1-gamma)*X)/L)**(1/epsilon)))

    return w


# def get_r(X, K, p_k, p):
#     '''
#     Parameters: Aggregate output, Aggregate capital

#     Returns:    Returns to capital
#     '''
#     #r = (alpha * (X / K)) - delta
#     r = ((p/p_k)*((A**((epsilon-1)/epsilon))*(((gamma*X)/K)**(1/epsilon)))*((1-((1-delta_tau)/(1-tau_g))*tau_b*delta_tau)**(-1.0)) - delta)*(1-tau_g)

#     return r


def solve_r(guess, K, X, p, p_k):
    '''
    This function solves for r - which is only defined implicitly in the dynamic firm problem with taxes
    '''

    r = guess 
    MPK = ((A**((epsilon-1)/epsilon))*(((gamma*X)/K)**(1/epsilon)))

    error = ((1-tau_b)*(p/p_k)*MPK)-delta - (r/(1-tau_g)) + (1-delta_tau)*tau_b*delta_tau*(((r/(1-tau_g))+delta)/((r/(1-tau_g))+delta_tau))

    #print 'diff in r due to delta_tau: ', (((1-tau_b)*(p/p_k)*MPK)-delta - (r/(1-tau_g)) + (1-delta_tau)*tau_b*delta_tau*(((r/(1-tau_g))+delta)/((r/(1-tau_g))+delta_tau)))-(((1-tau_b)*(p/p_k)*MPK)-delta - (r/(1-tau_g)) + (1-delta)*tau_b*delta*(((r/(1-tau_g))+delta)/((r/(1-tau_g))+delta)))

    #print ' print r solve for and error: ', r, error[0]
    #print ' all r errors: ', error

    return error[0]


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

def get_C(c_i):
    '''
    Parameters: c 

    Returns:    Aggregate consumption
    '''
    C = (np.tile(weights,(I,1,1))*c_i).sum(2).sum(1)    

    return C



def get_p(guesses, r, w):
    '''
    Generates price of producer output

    Returns: p (Mx1) vector of producer prices
    '''

    p = guesses 

    p_k = np.dot(xi,p)

    q = p_k*(((1-tau_d)/(1-tau_g))*(1-((1-delta_tau)*tau_b*delta_tau*((r+delta_tau)**(-1.0)))))

    k_over_x = get_k_over_x(p_k,p,r)
    l_over_x = get_l_over_x(p,w)

    #error = p - (w*l_over_x_vec + p_k*(r+delta)*k_over_x_vec)
    error = (p - (w*l_over_x + ((r*q)/((1-tau_d)*(1-tau_b)))*k_over_x + (((1-(tau_b*(1-delta_tau)))*delta*p_k)/(1-tau_b))*k_over_x))

    #print 'diff in q due to delta_tau: ', (p_k*(((1-tau_d)/(1-tau_g))*(1-((1-delta_tau)*tau_b*delta_tau*((r+delta_tau)**(-1.0))))))-(p_k*(((1-tau_d)/(1-tau_g))*(1-((1-delta)*tau_b*delta*((r+delta)**(-1.0))))))
    #print 'diff in p due to delta_tau: ', ((p - (w*l_over_x + ((r*q)/((1-tau_d)*(1-tau_b)))*k_over_x + (((1-(tau_b*(1-delta_tau)))*delta*p_k)/(1-tau_b))*k_over_x)))-((p - (w*l_over_x + ((r*q)/((1-tau_d)*(1-tau_b)))*k_over_x + (((1-(tau_b*(1-delta)))*delta*p_k)/(1-tau_b))*k_over_x)))
    mask = p < 0.0

    error[mask] = 1e14

    return error 


def get_k_over_x(p_k, p, r):
    '''
    Returns K/X for a firm
    Useful in determining price of output
    '''

    #k_over_x = gamma*(A**(epsilon-1))*(((p_k/p)*(r+delta))**(-1*epsilon))
    k_over_x = gamma*(A**(epsilon-1))*(((p_k/p)*((r/(1-tau_g))+delta)*(1-((1-delta_tau)*tau_b*delta_tau*(((r/(1-tau_g))+delta_tau)**(-1.0)))))**(-1*epsilon))

    #print 'diff in k_over_x due to delta_tau: ', (gamma*(A**(epsilon-1))*(((p_k/p)*((r/(1-tau_g))+delta)*(1-((1-delta_tau)*tau_b*delta_tau*(((r/(1-tau_g))+delta_tau)**(-1.0)))))**(-1*epsilon)))-(gamma*(A**(epsilon-1))*(((p_k/p)*((r/(1-tau_g))+delta)*(1-((1-delta)*tau_b*delta*(((r/(1-tau_g))+delta)**(-1.0)))))**(-1*epsilon)))
    return k_over_x

def get_l_over_x(p, w):
    '''
    Returns EL/X for a firm
    Useful in determining price of output
    '''

    l_over_x = (1-gamma)*(A**(epsilon-1))*((w/p)**(-1*epsilon))

    return l_over_x

def get_p_c(p):
    '''
    Generates price of consumption good

    Returns: p_c
    '''
    p_c = np.dot(pi,p)
    return p_c
    
def get_p_tilde(p_c):
    
    p_tilde = ((p_c/alpha)**alpha).prod()
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

def get_cons(w, r, n, k, bq, p_c, p_tilde, T_H, j):
    '''
    Parameters: Aggregate bequests by ability type

    Returns:    Bequests by age and ability
    '''
    k0 = np.zeros((S,1))
    k0[1:,0] = k[:-1,0] # capital start period with

    output = (((1+((1-tau_d[0])*r))*k0) + w*n*e[j] - k + bq + (T_H/weights.sum()) - ((p_c*cbar).sum()))/p_tilde
    #output = (((1+((1-tau_d[0])*r))*k0) + w*n*e[j] - k + bq - ((p_c*cbar).sum()))/p_tilde

    #print ' transfer used for consumption, ', T_H
    #print 'print diff in cons w and w/o transfer: ', np.absolute((((1+((1-tau_d[0])*r))*k0) + w*n*e[j] - k + bq + (T_H/weights.sum()) - ((p_c*cbar).sum()))/p_tilde -(((1+((1-tau_d[0])*r))*k0) + w*n*e[j] - k + bq - ((p_c*cbar).sum()))/p_tilde).max()

    return output
    

def get_k_demand(p_k,w,r,X):
    '''
    Parameters: Interest rate
                Output

    Returns:    Demand for capital by the firm
    '''
    #output = (gamma*X)/(((r+delta)**epsilon)*(A**(1-epsilon)))
    #output = (X/A)*(((gamma**(1/epsilon))+
    #          (((1-gamma)**(1/epsilon))*(((r+delta)*(p_k/w))**(epsilon-1))*
    #          (((1-gamma)/gamma)**((epsilon-1)/epsilon))))**(epsilon/(1-epsilon)))

    #output = (X/A)*(((gamma**(1/epsilon))+
    #          (((1-gamma)**(1/epsilon))*((p_k/w)**(epsilon-1))*((((r/(1-tau_g))+delta)*(1-(((1-delta_tau)/(1-tau_g))*tau_b*delta_tau)))**(epsilon-1))*
    #          (((1-gamma)/gamma)**((epsilon-1)/epsilon))))**(epsilon/(1-epsilon)))

    output = (X/A)*(((gamma**(1/epsilon))+
              (((1-gamma)**(1/epsilon))*((p_k/w)**(epsilon-1))*
              (((1-gamma)/gamma)**((epsilon-1)/epsilon))*((((r/(1-tau_g))+delta)*
              (1-((1-delta_tau)*tau_b*delta_tau*(((r/(1-tau_g))+delta_tau)**(-1.0)))))**(epsilon-1))))**(epsilon/(1-epsilon)))

    #print 'diff in k_demand due to delta_tau: ', ((X/A)*(((gamma**(1/epsilon))+(((1-gamma)**(1/epsilon))*((p_k/w)**(epsilon-1))*(((1-gamma)/gamma)**((epsilon-1)/epsilon))*((((r/(1-tau_g))+delta)*(1-((1-delta_tau)*tau_b*delta_tau*(((r/(1-tau_g))+delta_tau)**(-1.0)))))**(epsilon-1))))**(epsilon/(1-epsilon))))-((X/A)*(((gamma**(1/epsilon))+(((1-gamma)**(1/epsilon))*((p_k/w)**(epsilon-1))*(((1-gamma)/gamma)**((epsilon-1)/epsilon))*((((r/(1-tau_g))+delta)*(1-((1-delta)*tau_b*delta*(((r/(1-tau_g))+delta)**(-1.0)))))**(epsilon-1))))**(epsilon/(1-epsilon))))

    return output

def get_l_demand(p_k,w,r,K):
    '''
    Parameters: Wage rate
                Capital demand

    Returns:    Demand for labor by the firm
    '''
    #output = K*((1-gamma)/gamma)*(((r+delta)*(p_k/w))**epsilon)
    output = K*((1-gamma)/gamma)*((p_k/w)**epsilon)*((((r/(1-tau_g))+delta)*(1-((1-delta_tau)*tau_b*delta_tau*(((r/(1-tau_g))+delta_tau)**(-1.0)))))**epsilon)

    #print 'diff in l_demand due to delta_tau: ', (K*((1-gamma)/gamma)*((p_k/w)**epsilon)*((((r/(1-tau_g))+delta)*(1-((1-delta_tau)*tau_b*delta_tau*(((r/(1-tau_g))+delta_tau)**(-1.0)))))**epsilon))-(K*((1-gamma)/gamma)*((p_k/w)**epsilon)*((((r/(1-tau_g))+delta)*(1-((1-delta)*tau_b*delta*(((r/(1-tau_g))+delta)**(-1.0)))))**epsilon))

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
    error = (MUc(c[-1,:]))/p_tilde -  MUb(K_guess[-1, :])
    return error


def solve_hh(guesses, r, w, p_c, p_tilde, T_H, j):
    '''
    Parameters: SS interest rate (r), SS wage rate (w)
    Returns:    Savings (Sx1)
                Labor supply (Sx1)    

    '''
    k = guesses[0: S].reshape((S, 1))
    n = guesses[S:].reshape((S, 1))        
    BQ = get_BQ(r, k, j)
    bq = get_dist_bq(BQ,j)
    c = get_cons(w, r, n, k, bq, p_c, p_tilde, T_H, j)
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


# def solve_k(guesses, p, p_k, K_s, X):
#     K = guesses
#     numerator = ((p/p_k)*((gamma*(X/K))**(1/epsilon))*(A**((epsilon-1)/1))-delta)[0]
#     x_func = p_k*gamma*X*(((p_k/p)*(numerator+delta)*(A**((1-epsilon)/1)))**(-1*epsilon))
    
#     error = p_k*K-K_s+x_func.sum()-x_func

#     # Check and punish constraing violations
#     mask1 = K <= 0

#     error[mask1] = 1e14

#     #print 'solve k error: ', error
#     #print 'k_m guess: ', K
#     return error 

# def solve_l(guesses, p, L_s, X):
#     L = guesses
#     numerator = (p*(((1-gamma)*(X/L))**(1/epsilon))*(A**(epsilon-1)))[0]
#     x_func = (1-gamma)*X*(((numerator/p)*(A**((1-epsilon)/1)))**(-1*epsilon))
    
#     error = L-L_s+x_func.sum()-x_func

#     # Check and punish constraing violations
#     mask1 = L <= 0

#     error[mask1] = 1e14

#     #print 'solve l error: ', error
#     #print 'L_m guess: ', L
#     return error 



def solve_output(guesses,p_k,w,r,X_c):
    X = guesses
    Inv = np.reshape(delta*get_k_demand(p_k,w,r,X),(1,M)) # investment demand - will differ not in SS
    errors = np.reshape(X_c  + np.dot(Inv,xi) - X,(M))

    return errors

def Steady_State(guesses):
    '''
    Parameters: Steady state distribution of capital guess as array
                size SxJ and labor supply array of SxJ rss
    Returns:    Array of SxJ * 2 Euler equation errors
    '''
    
    r = guesses[0]
    w = guesses[1]
    T_H = guesses[2]

    # find prices of consumption and capital goods
    p_guesses = np.ones(M)
    p = opt.fsolve(get_p, p_guesses, args=(r, w), xtol=1e-9, col_deriv=1)
    p = p/p[0]
    p_c = get_p_c(p)
    p_tilde = get_p_tilde(p_c)
    p_k = np.dot(xi,p)


    #print 'prices ', p, p_c, p_k, p_tilde

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
        solutions = opt.fsolve(solve_hh, guesses, args=(r, w, p_c, p_tilde, T_H, j), xtol=1e-9, col_deriv=1)
        #out = opt.fsolve(solve_hh, guesses, args=(r, w, j), xtol=1e-9, col_deriv=1, full_output=1)
        #print'solution found flag', out[2], out[3]
        #solutions = out[0]
        k[:,j] = solutions[:S].reshape(S)
        n[:,j] = solutions[S:].reshape(S)
        BQ = get_BQ(r, k[:,j].reshape(S,1), j)
        bq = get_dist_bq(BQ, j).reshape(S,1)
        c[:,j] = get_cons(w, r, n[:,j].reshape(S,1), k[:,j].reshape(S,1), bq, p_c, p_tilde, T_H, j).reshape(S)

    c_i = ((p_tilde*np.tile(c,(I,1,1))*np.tile(np.reshape(alpha,(I,1,1)),(1,S,J)))/np.tile(np.reshape(p_c,(I,1,1)),(1,S,J)) 
                + np.tile(np.reshape(cbar,(I,1,1)),(1,S,J)))
    #print 'c_i', c_i

    # Find total consumption of each good
    C = get_C(c_i)
    #print 'total cons by good: ', C

    # Find total demand for output from each sector from consumption
    X_c = np.dot(np.reshape(C,(1,I)),pi)
    guesses = X_c/I
    x_sol = opt.fsolve(solve_output, guesses, args=(p_k, w, r, X_c), xtol=1e-9, col_deriv=1)

    X = x_sol
    #print 'Output by industry, ', X

    # find aggregate savings and labor supply
    K_s, K_constr = get_K(k)
    L_s = get_L(n)


    #### Need to solve for labor and capital demand from each industry
    K_d = get_k_demand(p_k, w, r, X)
    L_d = get_l_demand(p_k, w, r, K_d)

    #print 'Capital demands: ', K_d
    #print 'Labor demands: ', L_d

    # Find firm dividends
    DIV = ((1-tau_b)*(p*X - w*L_d) - ((1-(tau_b*(1-delta_tau)))*delta*p_k*K_d))
    #print 'diff in divs due to delta_tau: ', (((1-tau_b)*(p*X - w*L_d) - ((1-(tau_b*(1-delta_tau)))*delta*p_k*K_d)))-(((1-tau_b)*(p*X - w*L_d) - ((1-(tau_b*(1-delta)))*delta*p_k*K_d)))

    # Find total taxes paid
    firm_taxes = tau_b*(p*X-w*L_d) - tau_b*(1-delta_tau)*delta*p_k*K_d
    #print 'diff in firm taxes due to delta_tau: ', (tau_b*(p*X-w*L_d) - tau_b*(1-delta_tau)*delta*p_k*K_d)-(tau_b*(p*X-w*L_d) - tau_b*(1-delta)*delta*p_k*K_d)

    # Find value of each firm V = DIV/r in SS
    #V = (p*X - w*L_d - p_k*delta*K_d)/r
    V = ((1-tau_d)*DIV)/r

    print 'check V:', V.sum()-(p_k*K_d).sum()

    # Checking that interest rates are common across firms
    r_guess = r 
    r_implied = opt.fsolve(solve_r, r_guess, args=(K_d, X, p, p_k), xtol=1e-9, col_deriv=1)
    #print 'interest rates by firm and r guessed: ', r_implied, r

    # Alternative way to find factor demand from each industry as a function of factor supply
    # k_m_guesses = (X/X.sum())*K_s
    # l_m_guesses = (X/X.sum())*L_s
    # K_d_check = opt.fsolve(solve_k, k_m_guesses, args=(p, p_k, K_s, X), xtol=1e-9, col_deriv=1)
    # L_d_check = opt.fsolve(solve_l, l_m_guesses, args=(p, L_s, X), xtol=1e-9, col_deriv=1)


    # Check labor and asset market clearing conditions
    error1 = K_s - V.sum()
    error2 = L_s - L_d.sum()
    #error3 = T_H - (firm_taxes.sum() + (tau_d*DIV).sum())
    error3 = T_H - (firm_taxes.sum())
    

    print 'asset market diff: ', error1
    print 'labor market diff: ', error2
    print 'r, w: ', r, w
    print 'tax diff: ', error3
    print 'T_H: ', T_H

    # Check and punish violations
    if r <= 0:
        error1 += 1e9
    if r > 1:
        error1 += 1e9
    if w <= 0:
        error2 += 1e9

    #print 'r and w errors: ', error1, error2
    return [error1, error2, error3]
    

# Solve SS
r_guess_init = 0.97 #0.9 #0.746930316821
w_guess_init = 1.03 #2.5 #1.53867680151
T_H_guess_init = 0.0 #0.01  # total transfers, equal total tax rev here, tot pop here =1 so total equals per capita
guesses = [r_guess_init, w_guess_init, T_H_guess_init]
solutions = opt.fsolve(Steady_State, guesses, xtol=1e-12, col_deriv=1)
#solutions = Steady_State(guesses)
rss = solutions[0]
wss = solutions[1]
T_H_ss = solutions[2]
print 'ss r, w, T_H: ', rss, wss, T_H_ss


# find prices of consumption and capital goods
p_guesses = np.ones(M)
p_ss = opt.fsolve(get_p, p_guesses, args=(rss, wss), xtol=1e-9, col_deriv=1)
p_ss = p_ss/p_ss[0]
p_c_ss = get_p_c(p_ss)
p_tilde_ss = get_p_tilde(p_c_ss)
p_k_ss = np.dot(xi,p_ss)
print 'SS cons prices: ', p_ss, p_c_ss, p_k_ss, p_tilde_ss

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
    out = opt.fsolve(solve_hh, guesses, args=(rss, wss, p_c_ss, p_tilde_ss, T_H_ss, j), xtol=1e-9, col_deriv=1, full_output=1)
   # print'solution found flag', out[2], out[3]
    #print 'fsovle output: ', out[1]
    solutions = out[0]
    kss[:,j] = solutions[:S].reshape(S)
    nss[:,j] = solutions[S:].reshape(S)
    BQss = get_BQ(rss, kss[:,j].reshape(S,1), j)
    bqss = get_dist_bq(BQss, j).reshape(S,1)
    css[:,j] = get_cons(wss, rss, nss[:,j].reshape(S,1), kss[:,j].reshape(S,1), bqss, p_c_ss, p_tilde_ss, T_H_ss, j).reshape(S)
    # check Euler errors
    error1[:,j] = foc_k(rss, css[:,j].reshape(S,1), j).reshape(S-1) 
    error2[:,j] = foc_l(wss, nss[:,j].reshape(S,1), css[:,j].reshape(S,1), p_tilde_ss, j).reshape(S) 
    error3[:,j] = foc_bq(kss[:,j].reshape(S,1), css[:,j].reshape(S,1), p_tilde_ss)

c_i_ss = ((p_tilde_ss*np.tile(css,(I,1,1))*np.tile(np.reshape(alpha,(I,1,1)),(1,S,J)))/np.tile(np.reshape(p_c_ss,(I,1,1)),(1,S,J)) 
                + np.tile(np.reshape(cbar,(I,1,1)),(1,S,J)))

# Find total consumption of each good
C_ss = get_C(c_i_ss)
#print 'total cons by good: ', C_ss

# Find total demand for output from each sector from consumption
X_c_ss = np.dot(np.reshape(C_ss,(1,I)),pi)
guesses = X_c_ss/I
x_sol = opt.fsolve(solve_output, guesses, args=(p_k_ss, wss, rss, X_c_ss), xtol=1e-9, col_deriv=1)
X_ss = x_sol

# find aggregate savings and labor supply
K_s_ss, K_constr = get_K(kss)
L_s_ss = get_L(nss)

#### Need to solve for labor and capital demand from each industry
K_d_ss = get_k_demand(p_k_ss, wss, rss, X_ss)
L_d_ss = get_l_demand(p_k_ss, wss, rss, K_d_ss)


r_guess = rss 
r_implied_ss = r_implied = opt.fsolve(solve_r, r_guess, args=(K_d_ss, X_ss, p_ss, p_k_ss), xtol=1e-9, col_deriv=1)
print 'r diffs', rss-r_implied_ss

# Find value of each firm V = DIV/r in SS
DIV_ss = ((1-tau_b)*(p_ss*X_ss - wss*L_d_ss) - ((1-(tau_b*(1-delta_tau)))*delta*p_k_ss*K_d_ss))
V_ss = ((1-tau_d)*DIV_ss)/rss


# Check labor and asset market clearing conditions
asset_diff = K_s_ss - V_ss.sum()
labor_diff = L_s_ss - L_d_ss.sum()
print 'Market clearing diffs: ', asset_diff, labor_diff

firm_taxes_ss = tau_b*(p_ss*X_ss-wss*L_d_ss) - tau_b*(1-delta_tau)*delta*p_k_ss*K_d_ss
tax_diff = T_H_ss - (firm_taxes_ss.sum())
print 'Tax rev diffs: ', tax_diff


Yss = get_X(K_d_ss,L_d_ss)

Iss = np.reshape(delta*get_k_demand(p_k_ss, wss,rss,X_ss),(1,M)) # investment demand - will differ not in SS


print 'RESOURCE CONSTRAINT DIFFERENCE:'
print 'RC: ', X_ss - Yss
print 'RC2: ', X_ss - np.dot(pi.transpose(),C_ss)- np.dot(delta*xi.transpose(),K_d_ss)
print 'RC3: ', X_ss - np.dot(np.reshape(C_ss,(1,I)),pi)- np.dot(delta*K_d_ss,xi)
print 'RC4: ',  np.reshape(X_c_ss  + np.dot(Iss,xi) - X_ss,(M))

#print 'RC3: ', X_ss - pi[0,0]*C_ss[0] - pi[1,0]*C_ss[1]- delta*(xi[0,0]*K_d_ss[0] +xi[0,1]*K_d_ss[1]+xi[0,2]*K_d_ss[2])



print("Euler errors")
print(error1)
print(error2)
print(error3)

print 'kssmat: ', kss

print 'ss r, w, T_H: ', rss, wss, T_H_ss


