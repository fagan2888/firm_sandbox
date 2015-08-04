import numpy as np
from scipy import optimize
import sys

'''
Author: Rex McArthur
Last edited August 4, 2015

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
S = 10. #periods
J = 4.  #Ability groups
T = 25. #Time to Steady State  
A = 1.0 # Total Factor of Productivity 
bin_weights = [.25, .25, .25, .25] #Weights for ability gropus
gamma = .5 #Capital share of output  
epsilon = .6 #Elasticity of Substitution between capital and labor
delta = .02  #depreciation rate 
alpha = .2  #Preference parameter for two goods (1-alpha) is other 
sigma = 1.9 #relative risk averstion for hh
beta = .98  #Discount factor 
max_l = 1.0 #max hours 
cbar1 = 0.1 #min of product 1 
cbar2 = 0.1 #min of product 2   
chi_b = 0.2 #utility weight for bequest 
chi_n = 0.5 #disutility weight for labor

#Make an intial guess for r and w
r = .2
w = 2.0

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

def MUc(c):
    '''
    

    Marginal utility of Consumption
'''

p1 = get_p(r,w)
p2 = get_p(r,w)
print p1, p2
p_tilde = get_p_tilde(p1,p2)
print p_tilde




