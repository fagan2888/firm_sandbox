import numpy as np
from scipy import optimize
'''
_______Files that call this file_______
OLG2.py


____Files this function calls_____



_____Functions in this file_______
get_wage-   Parameters: capital and labor vectors
            Returns: Steady State wage
get_rate-   Parameters: capital and labor vectors
            Returns: Steady State rate
'''

def get_wage(kvec, lvec):
    '''
    Takes in vectors of S-1 capitals, S labors
    Returns the Steady State Wage 
    '''
    #TODO Redo the theory calculations on this function
    K = np.sum(kvec)
    L = np.sum(lvec)
    wage = (1-alpha)*A*(K/L)**alpha
    return wage

def get_rate(kvec, lvec):
    '''
    Takes in vectors of S-1 capitals, S labors
    Returns the Steady State Wage 
    '''
    #TODO Redo the theory calculations on this function
    K = np.sum(kvec)
    L = np.sum(lvec)
    rate = (alpha)*A*(L/K)**(1-alpha)
    return wage
    
def calc_ss(guess_vec):
    """
    This takes the Euler equations, and sets them equal to zero for an f-solve
    Remember that Keq was found by taking the derivative of the sum of the 
        utility functions, with respect to k in each time period, and that 
        leq was the same, but because l only shows up in 1 period, it has a
        much smaller term.

    ### Paramaters ###
    guess_vector: The first S-1 elements are the intial guess for the kapital
        The the last S are the intial guess for the labor
    """

