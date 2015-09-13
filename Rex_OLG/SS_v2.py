'''
Author: Rex McArthur
Last updated: 08/13/2015

Calculates Steady state OLG model with S age cohorts, J types, 2 static firms

'''
#Packages
import numpy as np
import scipy.optimize as opt
import time 
'''
Set up
______________________________________________________
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
______________________________________________________
'''

# Parameters
sigma = 1.9 # coeff of relative risk aversion for hh
beta = 0.98 # discount rate
alpha = np.array([0.29, 1.0-0.29]) # preference parameter - share of good i in composite consumption, shape =(I,), shares must sum to 1
cbar = np.array([0.000, 0.000]) # min cons of each of I goods, shape =(I,)
#delta = np.array([0.1, 0.1]) # depreciation rate
#delta = np.array([0.1, 0.12]) # depreciation rate
#delta = np.array([0.1, 0.12, 0.15]) # depreciation rate, shape =(M,)
delta = .1
A = 1.0 # Total factor productivity
#gamma = np.array([0.3, 0.25]) # capital's share of output
#gamma = np.array([0.3, 0.3]) # capital's share of output
#gamma = np.array([0.3, 0.25, 0.4]) # capital's share of output, shape =(M,)
gamma = .3
#xi = np.array([[0.2, 0.8],[0.3, 0.7]]) # fixed coeff input-output matrix
#pi = np.array([[0.5, 0.5],[0.1, 0.9]]) # fixed coeff pce-bridge matrix relating output and cons goods
#pi = np.array([[1.0, 0.0],[0.0, 1.0]]) # fixed coeff pce-bridge matrix relating output and cons goods
#xi = np.array([[0.2, 0.6, 0.2],[0.0, 0.2, 0.8], [0.6, 0.2, 0.2] ]) # fixed coeff input-output matrix, shape =(M,M)
xi = np.eye(2)
pi = np.eye(2)
#xi = np.array([[1.0, 0.0],[0.0, 1.0]]) # fixed coeff input-output matrix
#pi = np.array([[0.4, 0.3, 0.3],[0.1, 0.8, 0.1]]) # fixed coeff pce-bridge matrix relating output and cons goods, shape =(I,M)
#xi = np.array([[1.0, 0.0],[0.0, 1.0]]) # fixed coeff input-output matrix
#xi = np.array([[0.0, 1.0],[0.0, 1.0]]) # fixed coeff input-output matrix
#epsilon = np.array([0.6, 0.6]) # elasticity of substitution between capital and labor
#epsilon = np.array([0.55, 0.6, 0.62]) # elasticity of substitution between capital and labor, shape =(M,)
epsilon = .55
nu = 1.9 # elasticity of labor supply 
chi_n = 0.5 #utility weight, disutility of labor
chi_b = 0.2 #utility weight, warm glow bequest motive
ltilde = 1.0 # maximum hours
e = np.array([0.5, 1.0, 1.2, 1.7]) # effective labor units for the J types, shape =(J,)
#e = [1.0, 1.0, 1.0, 1.0] # effective labor units for the J types
S = 5 # periods in life of hh
J = 4 # number of lifetime income groups
I = 2 # number of consumption goods
M = 2 # number of production industries
#surv_rate = np.array([0.99, 0.98, 0.6, 0.4, 0.0]) # probability of surviving to next period, shape =(S,)
#surv_rate = np.array([1.0, 1.0, 1.0, 1.0, 0.0]) # probability of surviving to next period
#mort_rate = 1.0-surv_rate # probability of dying at the end of current period
#surv_rate[-1] = 0.0
#mort_rate[-1] = 1.0
#surv_mat = np.tile(surv_rate.reshape(S,1),(1,J)) # matrix of survival rates
#mort_mat = np.tile(mort_rate.reshape(S,1),(1,J)) # matrix of mortality rates
#surv_rate1 = np.ones((S,1))# prob start at age S
#surv_rate1[1:,0] = np.cumprod(surv_rate[:-1], dtype=float)
#omega = np.ones((S,J))*surv_rate1# number of each age alive at any time
#lambdas = np.array([0.5, 0.2, 0.2, 0.1])# fraction of each cohort of each type, shape =(J,)
#weights = omega*lambdas/((omega*lambdas).sum()) # weights - dividing so weights sum to 1

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


def get_r(X, K, p):
    '''
    Parameters: Aggregate output, Aggregate capital

    Returns:    Returns to capital
    '''
    #r = (alpha * (X / K)) - delta
    r = p*((A**((epsilon-1)/epsilon))*(((gamma*X)/K)**(1/epsilon))) - delta

    return r


def get_L(n):
    '''
    Parameters: n 

    Returns:    Aggregate labor
    '''
    L = np.sum((n*e))
    return L
    
def get_K(k):
    '''
    Parameters: k 

    Returns:    Aggregate capital
    '''
    K_constr = False
    K = np.sum(k)
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

def get_p(r, w):
    '''
    Generates price of consumption producer output

    Returns: p_c
    '''
    p = (((1-gamma)*((w/A)**(1-epsilon)))+(gamma*(((r+delta)/A)**(1-epsilon))))**(1/(1-epsilon))

    return p

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

def steady_state(guesses):
    '''
    Parameters: 
    Returns:
    '''
    r = guesses[0]
    w = guesses[1]

    #Find corresponding prices of consumption goods
    p_c1 = get_p(r,w)
    p_c2 = get_p(r,w)
    p = np.array((p_c1, p_c2))
    #If you want various prices, you have to add vectors for epsilon or delta or gamma
    #TODO REWRITE THIS IN TERMS OF TWO PRICES
    p_c = get_p_c(p)
    print p_c
    p_tilde = get_p_tilde(p_c)
    print p_tilde
    #p_c1 = 1
    #p_c2 = 1
    #p_tilde = 1

    k_guess = np.ones(S-1)*.05
    n_guess = np.ones(S)*.3
    guesses = np.zeros(2*S-1)
    guesses[:S]= n_guess
    guesses[S:] = k_guess

    #Solve the household problem in iterations of j
    k = np.zeros((S-1, J))
    n = np.zeros((S, J))
    c = np.zeros((S, J))
    guessvec = np.zeros(2*S-1)
    #TODO Here you could recursively make the previous problem be the new guess 
    #See Jason's code if you don't get it
    for j in xrange(J):
        guessvec = np.zeros((2*S-1))
        guessvec[:S] = n_guess[:]
        guessvec[S:] = k_guess[:]
        solutions = opt.fsolve(solve_house, guessevec, args = (r, w,
            p_c1, p_c2, p_tilde, 1), xtol = 1e-9, col_deriv = 1)
        raw_input('Hello')
        n[:,j]=solutions[:S].reshape(S)
        k[:,j] = solutions[S:].reshape(S)
        c[:,j] = get_cons(w,r,n[:,j].reshape(S,1), k[:,j].reshape(S,1), p_c, p_tilde, j).reshape(S)
    c_i = ((p_tilde*np.tile(c,(2,1,1))*np.tile(np.reshape(alpha,(2,1,1)),(1,S,J)))
            /np.tile(np.reshape(p_c,(2,1,1)),(1,S,J)) 
            + np.tile(np.reshape(cbar,(2,1,1)),(1,S,J)))

    #Total consumption for each good
    C = get_C(c_i)

    #print 'n, k ', new_n_vec, new_k_vec
    #new_K = get_K(new_k_vec)
    #new_L = get_L(new_n_vec)
    #new_Y = get_Y(new_K,new_L)
    #new_r = get_r(new_Y,new_K)
    #new_w = get_w(new_Y,new_L)
    #new_c = consump(new_w, new_r, new_n_vec, new_k_vec, p_c1, p_c2, p_tilde, 1)
    X_c = np.dot(np.reshape(C,(1,I)),pi)
    guesses = X_c/I
    x_sol = opt.fsolve(solve_output, guesses, args=(w, r, X_c), xtol=1e-9, col_deriv=1)

    X = x_sol

    # find aggregate savings and labor supply
    K_s, K_constr = get_K(k)
    L_s = get_L(n)

    # Find factor demand from each industry as a function of factor supply
    #K_d = K_s - get_sum_Xk(r,p,X)
    #L_d = L_s - get_sum_Xl(w,p,X)
    k_m_guesses = (X/X.sum())*K_s
    l_m_guesses = (X/X.sum())*L_s
    K_d = opt.fsolve(solve_k, k_m_guesses, args=(p, K_s, X), xtol=1e-9, col_deriv=1)
    L_d = opt.fsolve(solve_l, l_m_guesses, args=(p, L_s, X), xtol=1e-9, col_deriv=1)


    #### Need to solve for labor and capital demand from each industry
    K_d_check = get_k_demand(w, r, X)
    L_d_check = get_l_demand(w, r, K_d_check)

    ## Solve for factor demands in a third way
    #r_vec = np.array([r, r, r])
    #K_d_3 = K_s - (gamma*X*((((r_vec+delta)/p)*(A**((1-epsilon)/1)))**(-1*epsilon))).sum() - (gamma*X*((((r_vec+delta)/p)*(A**((1-epsilon)/1)))**(-1*epsilon))) 
    #print ' three k diffs: ', K_d-K_d_3, K_d-K_d_check, K_d_3-K_d_check

    # get implied factor prices
    r_new = get_r(X, K_d, p)[0]
    w_new = get_w(X, L_d, p)[0]
    #print 'all r_new values: ', get_r(X, K_d, p)
    #print 'all alt r_new values: ', get_r(X,K_d_check,p)
    #print 'alt r values: ', get_r(X,K_d_check,p)
    print 'diff btwn r: ', get_r(X, K_d, p) - get_r(X,K_d_check,p)
    print 'diff btwn k: ', K_d-K_d_check
    print 'diff btwn w: ', get_w(X, L_d, p) - get_w(X,L_d_check,p)
    print 'diff btwn l: ', L_d-L_d_check
    #print 'all w_new values: ', get_w(X, L_d, p)
    #print 'all alt w_new values: ', get_w(X,L_d_check,p)

    #print 'r diffs', r-get_r(X[0],K_d[0]), r-get_r(X[1],K_d[1])
    print 'market clearing: ', K_s - K_d.sum(),  L_s - L_d.sum()
    print 'market clearing 2: ', K_s - K_d_check.sum(), L_s - L_d_check.sum()

    # Check labor and capital market clearing conditions
    #error1 = K_s - K_d.sum()
    #error2 = L_s - L_d.sum()
    error1 = r_new - r
    error2 = w_new - w
    print 'errors: ', error1, error2
    print 'r, rnew, w, wnew: ', r, r_new, w, w_new

    # Check and punish violations
    if r <= 0:
        error1 += 1e9
    #if r > 1:
    #    error1 += 1e9
    if w <= 0:
        error2 += 1e9

    return [error1, error2]

    #TODO Add in punishing constraints
    print new_r, new_w
    return list((error1, error2))

r_guess_init = 0.77
w_guess_init = 1.03 
guesses = [r_guess_init, w_guess_init]
solutions = opt.fsolve(steady_state, guesses, xtol=1e-9, col_deriv=1)
#solutions = Steady_State(guesses)
rss = solutions[0]
wss = solutions[1]
print 'ss r, w: ', rss, wss

