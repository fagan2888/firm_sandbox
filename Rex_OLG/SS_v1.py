import numpy as np
from scipy import optimize
import sys

'''
Author: Rex McArthur
Last edited August 5, 2015

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


def consump(w, r, n, k0, k, p1, p2, p_tilde, j):
    print n.shape
    print k0.shape
    print k.shape

    c = (((1+r)*k0) + w*n*e[j] - k - (p1*cbar1) - (p2*cbar2)/p_tilde)
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
    kerror = MUc(c[:-1,0]) - (1+r)*beta*MUc(c[1:,0])
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
    print'n', n, n.shape
    k = guessvec[S:]
    print'k', k, k.shape

    k0 = np.zeros((S-1,1))
    print k0.shape
    print k0[1:,0].shape
    print k[:-1].shape
    k0[1:,0] = k[:-1]
    k0 = k0.flatten()
    print k0
    c = consump(w,r,n,k0,k,p1,p2,p_tilde,j)
    print c
    kerror = k_error(r,c,j)
    lerror = l_error(w,L_guess,c,p_tilde,j)

    #Duct Tape
    mask1 = n < 0
    mask2 = n > ltilde
    mask3 = k < 0
    mask4 = c <= 0
    lerror[mask1] += 1e10
    lerror[mask2] += 1e10
    kerror[mask3[:-1,0]] += 1e10
    kerror[mask4[:-1,0]] += 1e10
    
    totalerror =  list(kerror)+list(lerror)
    print totalerror
    return totalerror


#Make an intial guess for r and w
rguess = .2
wguess = 1.3
kguess = .1
nguess = .4

p1 = get_p(rguess,wguess)
p2 = get_p(rguess,wguess)
print p1, p2
p_tilde = get_p_tilde(p1,p2)
print p_tilde
guessvec = np.ones(2*S-1)
guessvec[:S] = nguess
guessvec[S:] = kguess


for j in xrange(J):
    x = optimize.fsolve(solve_house, guessvec, args =(rguess, wguess, p1, p2, p_tilde, j))


