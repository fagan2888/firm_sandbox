import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
import cPickle as pkl

'''
This is the main executable file


_______Files this function calls________
SS_func.py  - contains all the functions for the steady state calculations
    get_wage- Paramaters: kvec, lvec
    get_rate-
TPI_func.py - contains all the functions for the tpi calculations

__________Functions in this file_______

_____________Paramaters________________
beta     = discount factor (0,1)**years per period
delta    = depreciation, (0,1)**years per period
gamma    = consumption and labor relative risk averstion, <1
alpha    = cobb-douglas ouput elasticity of labor (0,1)
A        = Firm productivity coeffecient >0
S        = number of periods you wish to use (3,80)
years    = number of years per period (1,20)
error    = Intializes an error for the TPI >5
xi       = Used to calculate your convex combonation for TPI
epsilon  = Accuracy parameter in TPI calculation
T        = Number of periods to reach the steady state, should be large
shock    = How much to shock the economy away from the Steady State

guess_vector = intial guess for the cap fsolve. The first s entries are capital guess, the last s entries are labor guesses
show_graph = boolean variable, plots graph if True
'''

# Paramaters
s = 20.
years = 60./s
beta = .96**years
delta = 1-(1-.05)**years
gamma = 2.9
alpha = .35
A = 1.
xi = .5
epsilon = 10e-9
T = 70
shock = 1.1
labor_guess = .8
cap_guess = .05
error = 1
show_graph = True







