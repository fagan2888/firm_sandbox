import numpy as np
from scipy import optimize
import sys

'''
Author: Rex McArthur
Last edited August 10, 2015

This is intended to be a explicitly typed two firm steady state solver.
Firms will differentiate with different epsilon values, and productivity
values. General methodolgy will follow computation outlined in the document
entitled adding firms in the repo. Includes endogenous labor, bequest 
motives.
'''

'''
_______Parameters______
S           = Periods
J           = Ability groups
T           = Time to Steady State
A           = Total factor of Productivity
bin_weights = Share of agents in each ability group
gamma       = capital share of output
epsilon     = Elasticity of substitution between capital and labor
delta       = depreciation rate
alpha       = preference parameter for each good
sigma       = Relative risk aversion
beta        = discount rate
max_l       = max amount of hours to be worked
cbar1       = min of product 1 in composite good
cbar2       = min of product 2 in composite good
chi_b       = utility weight warm glow
chi_n       = utility weight disutility of labor
ltilde      = Max labor endowed
'''
S = 10 #periods
J = 4  #Ability groups
T = 25 #Time to Steady State  
A = 1.0 # Total Factor of Productivity 
e = [.8, 1.0, 1.5, 1.2]
bin_weights = [.25, .25, .25, .25] #Weights for ability gropus
gamma = .5 #Capital share of output  
epsilon = .6 #Elasticity of Substitution between capital and labor
delta = .02  #depreciation rate 
alpha = .2  #Preference parameter for two goods (1-alpha) is other 
sigma = 1.9 #relative risk averstion for hh
beta = .98  #Discount factor 
max_l = 1.0 #max hours 
cbar1 = 0. #min of product 1 
cbar2 = 0. #min of product 2   
chi_b = 0.2 #utility weight for bequest 
chi_n = 0.5 #disutility weight for labor
ltilde = 1. #Max labor endowment 
nu = 2.0


#Logical checks
if np.sum(bin_weights) != 1.:
    print 'ERROR: Ability weights not equal to one'
    sys.exit(0) 
if [i for i in bin_weights if i <0]:
    print 'ERROR: Ability weights less than 0'
    sys.exit(0)

def get_p(r,w):
    '''
    Equation 23 
    Returns the price for the firm's good, given a guess for r, w
    '''
    p = ((1-gamma)*((w/A)**(1-epsilon))+(gamma*(((r+delta)/A)**(1-epsilon))))**(1/(1-epsilon))
    return p

def get_p_tilde(p1, p2):
    '''
    Equation 25
    Price of composite good
    '''
    p_tilde = ((p1/alpha)**alpha)*((p2/(1-alpha))**(1-alpha))
    return p_tilde

def get_K(k):
    '''
    Returns market capital by summing individual capital
    '''
    K = np.sum(k)
    return K

def get_L(l):
    '''
    Returns market labor by summing individual labor
    '''
    L = np.sum(l)
    return L

def get_Y(K,L):
    '''
    Returns total output
    '''
    Y = (K**alpha) * L**(1-alpha)
    return Y

def get_r(Y,K):
    '''
    Returns the interest rate determined by the Capital Stock and Output
    '''
    r = (alpha *(Y/K))-delta
    return r

def get_w(Y,L):
    '''
    returns the wage determined by the output and Labor stock
    '''
    w = (1-alpha)*Y/L
    return w

def consump(w, r, n, k, p1, p2, p_tilde, j):
    '''
    returns S long vector of consumption for each period of life
    '''
    k_0 = np.zeros(S)
    k_0[:-1] = k
    k_1 = np.zeros(S)
    k_1[:-1]=k
    #print n.shape
    #print k_0.shape
    #print k_1.shape

    c = (((1+r)*k_0) + w*n*e[j] - k_1 - (p1*cbar1) - (p2*cbar2))/p_tilde
    return c

def MUc(c):
    '''
    Marginal utility of Consumption
    '''
    mu_c = c**(-sigma)
    return mu_c

def MUl(n):
    '''
    Marginal (dis)utility of labor
    '''
    mu_l = -chi_n*((ltilde-n)**(-nu))
    return mu_l

def k_error(r,c,j):
    '''
    Parameters:
    w = wage
    r = rate
    c = consumption array (SxJ)
    
    returns the k error for the inner fsolve
    '''
    kerror = MUc(c[:-1]) - (1+r)*beta*MUc(c[1:])
    return kerror

def l_error(w, L_guess, c, p_tilde, j):
    '''
    w = wage
    r = rental rate
    L_guess = labor array (SxJ)
    j = ability type weights

    returns the l error for the inner fsolve
    '''
    lerror = (w*MUc(c)*e[j])/p_tilde + MUl(L_guess)
    return lerror

def solve_house(guessvec, r, w, p1, p2, p_tilde, j):
    '''
    '''
    n = guessvec[0:S]
    #print'n', n, n.shape
    k = guessvec[S:]
    #print 'size k: ', k.shape
    #print'k', k, k.shape

    #k0 = np.zeros((S-1,1))
    #print k0.shape
    #print k0[1:,0].shape
    #print k[:-1].shape
    #k0[1:,0] = k[:-1]
    #k0 = k0.flatten()
    #print k0
    K = get_K(k)
    L = get_L(n) 
    Y = get_Y(K,L)
    r = get_r(Y,K) 
    w = get_w(Y,L)
    c = consump(w,r,n,k,p1,p2,p_tilde,j)
    #print c
    kerror = k_error(r,c,j)
    lerror = l_error(w,n,c,p_tilde,j)

    #Duct Tape
    mask1 = n < 0
    mask2 = n > ltilde
    mask3 = k < 0
    #print 'size mask3: ', mask3.shape
    #print 'size kerror: ', kerror.shape
    mask4 = c <= 0
    lerror[mask1] += 1e10
    lerror[mask2] += 1e10
    #kerror[mask3[:-1]] += 1e10
    #kerror[mask4[:-1]] += 1e10
    kerror[mask3] += 1e10
    kerror[mask4[:-1]] += 1e10
    
    totalerror =  list(kerror)+list(lerror)
    #print totalerror
    #print 'labors guess: ', n
    #print 'savings guess: ', k
    #print 'cons guess: ', c
    print 'max euler error: ', np.absolute(np.asarray(totalerror)).max()
    return totalerror

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
    p_tilde = get_p_tilde(p_c1, p_c2)
    #p_c1 = 1
    #p_c2 = 1
    #p_tilde = 1

    k_guess = np.ones(S-1)*.05
    n_guess = np.ones(S)*.3
    guesses = np.zeros(2*S-1)
    guesses[:S]= n_guess
    guesses[S:] = k_guess

    
    solutions = optimize.fsolve(solve_house, guesses, args = (r, w,
        p_c1, p_c2, p_tilde, 1), xtol = 1e-9, col_deriv = 1)
    new_n_vec = solutions[:S] 
    new_k_vec = solutions[S:]
    print 'n, k ', new_n_vec, new_k_vec
    new_K = get_K(new_k_vec)
    new_L = get_L(new_n_vec)
    new_Y = get_Y(new_K,new_L)
    new_r = get_r(new_Y,new_K)
    new_w = get_w(new_Y,new_L)
    new_c = consump(new_w, new_r, new_n_vec, new_k_vec, p_c1, p_c2, p_tilde, 1)
    return list((r-new_r, w-new_w))

def solve(guesses):

    x = optimize.fsolve(steady_state, guessvec)
    r_ss = x[0]
    w_ss = x[1]
    print 'rate ', r_ss
    print 'wage ', w_ss
    r = guesses[0]
    w = guesses[1]

    #Find corresponding prices of consumption goods
    p_c1 = get_p(r,w)
    p_c2 = get_p(r,w)
    p_tilde = get_p_tilde(p_c1, p_c2)
    #p_c1 = 1
    #p_c2 = 1
    #p_tilde = 1

    k_guess = np.ones(S-1)*.05
    n_guess = np.ones(S)*.3
    guesses = np.zeros(2*S-1)
    guesses[:S]= n_guess
    guesses[S:] = k_guess

    solutions = optimize.fsolve(solve_house, guesses, args = (r, w,
        p_c1, p_c2, p_tilde, 1), xtol = 1e-9, col_deriv = 1)
    new_n_vec = solutions[:S] 
    new_k_vec = solutions[S:]
    print 'n, k ', new_n_vec, new_k_vec
    new_K = get_K(new_k_vec)
    new_L = get_L(new_n_vec)
    new_Y = get_Y(new_K,new_L)
    new_r = get_r(new_Y,new_K)
    new_w = get_w(new_Y,new_L)
    new_c = consump(new_w, new_r, new_n_vec, new_k_vec, p_c1, p_c2, p_tilde, 1)
    return new_c
    

#Make an intial guess for r and w
rguess =  (1/beta)-1
wguess = 0.55
kguess = .1
nguess = .4

guessvec = np.array((rguess,wguess))
x = optimize.fsolve(steady_state, guessvec)
r_ss = x[0]
w_ss = x[1]
print 'rate ', r_ss
print 'wage ', w_ss
p1_ss = get_p(r_ss, w_ss)
p2_ss = get_p(r_ss, w_ss)
p_tilde_ss = get_p_tilde(p1_ss, p2_ss)
print 'Prices: ', p1_ss, p2_ss, p_tilde_ss
c1ss






