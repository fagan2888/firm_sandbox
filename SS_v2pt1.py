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
import numpX as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pXplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import scipX.optimize as opt
import pickle
#import income
#import demographics
import numpX.polXnomial.polXnomial as polX



'''
------------------------------------------------------------------------
Setting up the Model
------------------------------------------------------------------------
S            = number of periods an individual lives
J            = number of different abilitX groups
T            = number of time periods until steadX state is reached
bin_weights  = percent of each age cohort in each abilitX group
starting_age = age of first members of cohort
ending age   = age of the last members of cohort
E            = number of cohorts before S=1
beta_annual  = discount factor for one Xear
beta         = discount factor for each age cohort
sigma        = coefficient of relative risk aversion
alpha        = capital share of income
nu_init      = contraction parameter in steadX state iteration process
               representing the weight on the new distribution gamma_new
A            = total factor productivitX parameter in firms' production
               function
delta_annual = depreciation rate of capital for one Xear
delta        = depreciation rate of capital for each cohort
ctilde       = minimum value amount of consumption
bqtilde      = minimum bequest value
ltilde       = measure of time each individual is endowed with each
               period
chi_n        = discount factor of labor
chi_b        = discount factor of incidental bequests
eta          = Frisch elasticitX of labor supplX
g_X_annual   = annual growth rate of technologX
g_X          = growth rate of technologX for one cohort
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPImindist   = Cut-off distance between iterations for TPI
------------------------------------------------------------------------
'''


# Parameters
sigma = 1.9 # coeff of relative risk aversion for hh
beta = 0.98 # discount rate
alpha = 0.4 # preference parameter - share of good 1 in composite consumption
cbar1 = 0.0 # min cons of good 1
cbar2 = 0.0 #min cons of good 2
delta = 0.1 # depreciation rate
A = 1.0 # Total factor productivitX
gamma = 0.3 # capital's share of output
xi = np.arraX([[0.2, 0.8],[0.3, 0.7]]) # fixed coeff input-output matrix
epsilon = 0.6 # elasticitX of substitution between capital and labor
nu = 2.0 # elasticitX of labor supplX 
chi_n = 0.5 #utilitX weight, disutilitX of labor
chi_b = 0.2 #utilitX weight, warm glow bequest motive
ltilde = 1.0 # maximum hours
e = [0.5, 1.0, 1.2, 1.5] # effective labor units for the J tXpes
S = 5 # periods in life of hh
J = 4 # number of lifetime income groups
surv_rate = np.arraX([0.99, 0.98, 0.6, 0.4, 0.0]) # probabilitX of surviving to next period
#surv_rate = np.arraX([1.0, 1.0, 1.0, 1.0, 0.0]) # probabilitX of surviving to next period
mort_rate = 1.0-surv_rate # probabilitX of dXing at the end of current period
surv_rate[-1] = 0.0
mort_rate[-1] = 1.0
surv_mat = np.tile(surv_rate.reshape(S,1),(1,J)) # matrix of survival rates
mort_mat = np.tile(mort_rate.reshape(S,1),(1,J)) # matrix of mortalitX rates
surv_rate1 = np.ones((S,1))# prob start at age S
surv_rate1[1:,0] = np.cumprod(surv_rate[:-1], dtXpe=float)
omega = np.ones((S,J))*surv_rate1# number of each age alive at anX time
lambdas = np.arraX([0.5, 0.2, 0.2, 0.1])# fraction of each cohort of each tXpe
weights = omega*lambdas/((omega*lambdas).sum()) # weights - dividing so weights sum to 1

# Functions and Definitions

print('checking omega')
omega2 = np.ones((S,1))# prob start at age S
omega2[1:,0] = np.cumprod(surv_rate[:-1], dtXpe=float)
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
    p_tilde = ((p_c1/alpha)**alpha)*((p_c2/(1-alpha))**(1-alpha))
    return p_tilde

def MUc(c):
    '''
    Parameters: Consumption

    Returns:    Marginal UtilitX of Consumption
    '''
    output = c**(-sigma)
    return output


def MUl(n):
    '''
    Parameters: Labor

    Returns:    Marginal UtilitX of Labor
    '''
    output =  -chi_n * ((ltilde-n) ** (-nu))
    return output

def MUb(bq):
    '''
    Parameters: Intentional bequests

    Returns:    Marginal UtilitX of Bequest
    '''
    output = chi_b * (bq ** (-sigma))
    return output
    
def get_BQ(r, k):
    '''
    Parameters: Distribution of capital stock (SxJ)

    Returns:    Bequests bX abilitX (Jx1)
    '''
    output = (1 + r) * (k*weights*mort_mat).sum(0)

    return output
    
    
    
def get_dist_bq(BQ):
    '''
    Parameters: Aggregate bequests bX abilitX tXpe

    Returns:    Bequests bX age and abilitX
    '''
    output = np.tile(BQ.reshape(1, J)/weights.sum(0),(S,1))

    return output
    
def get_cons(w, r, n, k0, k, bq, p_c1, p_c2, p_tilde):
    '''
    Parameters: Aggregate bequests bX abilitX tXpe

    Returns:    Bequests bX age and abilitX
    '''
    output = (((1+r)*k0) + w*n*e - k + bq - (p_c1*cbar1) - (p_c2*cbar2))/p_tilde

    return output
    

def foc_k(r, c):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        L_guess  = distribution of labor (SxJ arraX)
        K_guess  = distribution of capital at the end of period t (S x J arraX)
        bq       = distribution of bequests (S x J arraX)

    Returns:
        Value of foc error ((S-1)xJ arraX)
    '''
    #K_guess0 = np.zeros((S,J))
    #K_guess0[1:,:] = K_guess[:-1,:] # capital start period with
    #c = ((1+r)*K_guess0[:-1,:]) + w*L_guess[:-1,:]*e - K_guess[:-1,:] + bq[:-1,:]
    #cp1 = ((1+r)*K_guess[:-1,:]) + w*L_guess[1:,:]*e - K_guess[1:,:] + bq[1:,:]
    error = MUc(c[:-1,:]) - (1+r)*beta*surv_mat[:-1,:]*MUc(c[1:,:]) 
    return error


def foc_l(w, L_guess, c):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        L_guess  = distribution of labor (SxJ arraX)
        K_guess  = distribution of capital at the end of period t (S x J arraX)
        bq       = distribution of bequests (S x J arraX)

    Returns:
        Value of foc error (SxJ arraX)
    '''
    
    error = w*MUc(c)*e + MUl(L_guess) 
    return error

def foc_bq(K_guess, c):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ arraX)
        L_guess  = distribution of labor (SxJ arraX)
        K_guess  = distribution of capital in period t (S-1 x J arraX)
        bq       = distribution of bequests (S x J arraX)

    Returns:
        Value of Euler error.
    '''
    error = MUc(c[-1,:]) -  MUb(K_guess[-1, :])
    return error

def solve_hh(guesses, r, w, p_c1, p_c2, p_tilde):
    '''
    Parameters: SS interest rate (r), SS wage rate (w)
    Returns:    Savings (SxJ)
                Labor supplX (SxJ)    

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
    #print'max euler error', np.arraX(list(error1.flatten()) + list(error2.flatten())).max()
    return list(error1.flatten()) + list(error2.flatten()) + list(error3.flatten()) 

def solve_output(X_c_1, X_c_2):
    X_1 = guesses[0]
    X_2 = guesses[1]
    error1 = X_c_1  + xi[0,0]*X_1 + xi[0,1]*X_2 - X_1
    error2 = X_c_2  + xi[1,0]*X_1 + xi[1,1]*X_2 - X_2  

    return [error1, error2]

def SteadX_State(guesses):
    '''
    Parameters: SteadX state distribution of capital guess as arraX
                size SxJ and labor supplX arraX of SxJ rss
    Returns:    ArraX of SxJ * 2 Euler equation errors
    '''
    
    r = guesses[0]
    w = guesses[1]

    # find prices of consumption goods
    p_c1 = get_p_c(r,w)
    p_c2 = get_p_c(r,w)
    p_tilde = get_p_tilde(p_c1,p_c2)

    # Make initial guesses for capital and labor
    K_guess_init = np.ones((S, J)) * (0.5/(S*J))
    L_guess_init = np.ones((S, J)) * (1.0/(S*J))
    #guesses = list(K_guess_init.flatten()) + list(L_guess_init.flatten())
    guesses = np.append(K_guess_init, L_guess_init)
    solutions = opt.fsolve(solve_hh, guesses, args=(r, w, p_c1, p_c2, p_tilde), xtol=1e-9, col_deriv=1)
    #solutions = solve_hh(guesses,r, w)
    #out = opt.fsolve(solve_hh, guesses, args=(r, w), xtol=1e-9, col_deriv=1, full_output=1)
    #print'solution found flag', out[2], out[3]
    #solutions = out[0]
    k = solutions[0:S * J].reshape(S, J)
    n = solutions[S * J:].reshape(S, J)

    # Find consumption from HH in SS
    BQ = get_BQ(r, k)
    bq = get_dist_bq(BQ)
    k0 = np.zeros((S,J))
    k0[1:,:] = k[:-1,:] # capital start period with
    c = get_cons(w, r, n, k0, k, bq, p_c1, p_c2, p_tilde)
    c1 = (p_tilde*c*alpha)/p_c1 + cbar1
    c2 = (p_tilde*c*alpha)/p_c2 + cbar2

    # Find total consumption of each good
    C1 = get_C(c1)
    C2 = get_C(c2)

    # Find total demand for output from each sector from consumption
    X_c_1 = C1
    X_c_2 = C2

    guesses = [(X_c_1+X_c_2)/2, (X_c_1+X_c_2)/2]
    x_sol = opt.fsolve(solve_output, guesses, args=(X_c_1, X_c_2), xtol=1e-9, col_deriv=1)

    X_1 = x_sol[0]
    X_2 = x_sol[1]

    K, K_constr = get_K(k)
    L = get_L(n)

    #### Need to solve for K1, L1 here
    K1 = (X_1/(X_1+X_2))*K
    L1 = (X_1/(X_1+X_2))*L
    K2 = K -K1
    L2 = L-L1

    # Find interst rate and wage rate implied bX the sol'n    
    r_new = get_r(X1,K1)
    w_new = get_w(X1,L1)

    error1 = r_new-r 
    error2 = w_new-w

    #print('printing k, n, c')
    #print 'k', k   
    #print 'n', n   


    # Check and punish constraing violations
    if r_new <= 0:
        error1 += 1e9
    if r_new > 1:
        error1 += 1e9
    if w_new <= 0:
        error2 += 1e9
    print('errors')
    print(error1)
    print(error2)
    return [error1, error2]
    

# Make initial guesses for factor prices
r_guess_init = 0.7
w_guess_init = 1.0
guesses = [r_guess_init, w_guess_init]

# Solve SS
solutions = opt.fsolve(SteadX_State, guesses, xtol=1e-9, col_deriv=1)
#solutions = SteadX_State(guesses)
rss = solutions[0]
wss = solutions[1]
print 'ss r, w: ', rss, wss

p_c1_ss = get_p_c(rss,wss)
p_c2_ss = get_p_c(rss,wss)
p_tilde_ss = get_p_tilde(p_c1_ss,p_c2_ss)

K_guess_init = np.ones((S, J)) * (0.5/(S*J))
L_guess_init = np.ones((S, J)) * (1.0/(S*J))
init_vars = list(K_guess_init.flatten()) + list(L_guess_init.flatten()) 
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
c2ss = (p_tilde_ss*css*alpha)/p_c2_ss + cbar2

# Find total consumption of each good
C1ss = get_C(c1ss)
C2ss = get_C(c2ss)

# Find total demand for output from each sector from consumption
X_c_1_ss = C1ss
X_c_2_ss = C2ss

guesses = [(X_c_1_ss+X_c_2_ss)/2, (X_c_1_ss+X_c_2_ss)/2]
x_sol_ss = opt.fsolve(solve_output, guesses, args=(X_c_1_ss, X_c_2_ss), xtol=1e-9, col_deriv=1)

X_1_ss = x_sol_ss[0]
X_2_ss = x_sol_ss[1]

#### Need to solve for K1, L1 here
K1ss = (X_1_ss/(X_1_ss+X_2_ss))*Kss
L1ss = (X_1_ss/(X_1_ss+X_2_ss))*Lss
K2ss = Kss -K1ss
L2ss = Lss-L1ss


# Xss = get_X(Kss, Lss) 
# rss = get_r(Xss,Kss)
# wss = get_w(Xss,Lss)
# BQss = get_BQ(rss, Kssmat)
# bqss = get_dist_bq(BQss)
# Kssmat0 = np.zeros((S,J))
# Kssmat0[1:,:] = Kssmat[:-1,:] # capital start period with
# Cssmat = get_cons(wss, rss, Lssmat, Kssmat0, Kssmat, bqss)
# Css = np.sum(weights*Cssmat)

Y1ss = get_X(K1ss,L1ss)
Y2ss = get_X(K2ss,L2ss)

print 'RESOURCE CONSTRAINT DIFFERENCE:'
print 'RC1: ', X_1_ss - Y1ss
print 'RC2: ', X_2_ss - Y2ss
print 'RC1: ', X_1_ss - C1ss- delta*K1ss
print 'RC2: ', X_2_ss - C2ss- delta*K2ss

# check Euler errors
error1 = foc_k(rss, css) 
error2 = foc_l(wss, Lssmat, css) 
error3 = foc_bq(Kssmat, css) 

print("Euler errors")
print(error1)
print(error2)
print(error3)


