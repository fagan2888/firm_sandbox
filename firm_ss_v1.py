'''
------------------------------------------------------------------------
Last updated 7/15/2015

This program runs the steady state solver to solve for a simple OG model
with exogenous labor supply and representative firms in two production industries 


This Python script calls the following other file(s) with the associated
functions:
    firm_funcs_v1.py
        get_r
        get_w
    hh_funcs_v1.py
        get_cvec_ss
        get_b_errors
        EulerSys
    agg_funcs_v1.py
        feasible
        get_L
        get_K
        get_Y
        get_C
        SS
------------------------------------------------------------------------
'''
# Import packages

import numpy as np
import agg_funcs_v1 as agg
reload(agg)

'''
------------------------------------------------------------------------
Declare parameters
------------------------------------------------------------------------
S            = integer in [3,80], number of periods an individual lives
T            = integer > S, number of time periods until steady state
beta_annual  = scalar in [0,1), discount factor for one year
beta         = scalar in [0,1), discount factor for each model period
sigma        = scalar > 0, coefficient of relative risk aversion
alpha        = scalar in (0,1)
A            = scalar > 0, total factor productivity parameter in firms'
               production function
gamma        = scalar in (0,1), capital share of income
delta_annual = scalar in [0,1], one-year depreciation rate of capital
delta        = scalar in [0,1], model-period depreciation rate of
               capital
SS_tol       = scalar > 0, tolerance level for steady-state fsolve
SS_graphs    = boolean, =True if want graphs of steady-state objects
TPI_solve    = boolean, =True if want to solve TPI after solving SS
TPI_tol      = scalar > 0, tolerance level for fsolve's in TPI
maxiter_TPI  = integer >= 1, Maximum number of iterations for TPI
mindist_TPI  = scalar > 0, Convergence criterion for TPI
xi           = scalar in (0,1], TPI path updating parameter
TPI_graphs   = boolean, =True if want graphs of TPI objects
------------------------------------------------------------------------
'''
# Household parameters
S = int(80)
T = int(round(2.5 * S))
beta_annual = .96
beta = beta_annual ** (80 / S)
sigma = 3.0
alpha = 0.3
cbar1 = 0.001
cbar2 = 0.002
# Firm parameters
A = 1.0
gamma = 0.35
epsilon = 0.7
delta_annual = .05
delta = 1 - ((1-delta_annual) ** (80 / S))
# SS parameters
SS_tol = 1e-13
SS_graphs = False
# TPI parameters
# TPI_solve = True
# TPI_tol = 1e-13
# maxiter_TPI = 100
# mindist_TPI = 1e-13
# xi = .20
# TPI_graphs = True

'''
------------------------------------------------------------------------
Compute the steady state
------------------------------------------------------------------------
b_guess       = [S-1,] vector, initial guess for steady-state
                distribution of savings
feas_params   = [4,] vector, parameters for feasible function
                [S, A, alpha, delta]
GoodGuess     = boolean, =True if initial steady-state guess is feasible
K_constr_init = boolean, =True if K<=0 for initial guess b_guess
c_constr_init = [S,] boolean vector, =True if c<=0 for initial b_guess
ss_params     = [7,] vector, parameters to be passed in to SS function
b_ss          = [S-1,] vector, steady-state distribution of savings
c_ss          = [S,] vector, steady-state distribution of consumption
w_ss          = scalar > 0, steady-state real wage
r_ss          = scalar > 0, steady-state real interest rate
K_ss          = scalar > 0, steady-state aggregate capital stock
EulErr_ss     = [S-1,] vector, steady-state Euler errors
L_ss          = scalar > 0, steady-state aggregate labor
Y_params      = [2,] vector, production function parameters [A, alpha]
Y_ss          = scalar > 0, steady-state aggregate output (GDP)
C_ss          = scalar > 0, steady-state aggregate consumption
rcdiff_ss     = scalar, steady-state difference in goods market clearing
                (resource constraint)
------------------------------------------------------------------------
'''
# Make initial guess of the steady-state
b_guess = 0.1 * np.ones(S-1)
# # Make sure initial guess is feasible
# feas_params = np.array([S, A, alpha, delta])
# GoodGuess, K_constr_init, c_constr_init = s2f.feasible(feas_params, b_guess)
# if K_constr_init == True and c_constr_init.max() == False:
#     print 'Initial guess is not feasible because K<=0. Some element(s) of bvec must increase.'
# elif K_constr_init == False and c_constr_init.max() == True:
#     print 'Initial guess is not feasible because some element of c<=0. Some element(s) of bvec must decrease.'
# elif K_constr_init == True and c_constr_init.max() == True:
#     print 'Initial guess is not feasible because K<=0 and some element of c<=0. Some element(s) of bvec must increase and some must decrease.'
# elif GoodGuess == True:
    # print 'Initial guess is feasible.'

    # Compute steady state
print 'BEGIN STEADY STATE COMPUTATION' 
ss_params = np.array([S, beta, sigma, alpha, cbar1, cbar2, A, gamma, epsilon, delta, SS_tol])
b_ss, EulErr_ss, C1_ss, C2_ss, Y1_ss, Y2_ss, K_ss, K1_ss, K2_ss  = agg.SS(ss_params, b_guess, SS_graphs)

#Print diagnostics
print 'The maximum absolute steady-state Euler error is: ', np.absolute(EulErr_ss).max()
    # print 'The steady-state distribution of capital is:'
#print b_ss
    # print 'The steady-state distribution of consumption is:'
    # print c_ss
    # print 'The steady-state wage, interest rate, and aggregate capital are:'
    # print np.array([w_ss, r_ss, K_ss])
    # L_ss = s2f.get_L(np.ones(S))
    # Y_params = np.array([A, alpha])
    # Y_ss = s2f.get_Y(Y_params, K_ss, L_ss)
    # C_ss = s2f.get_C(c_ss)
rcdiff_ss = Y2_ss - C2_ss - delta * (K_ss-((Y1_ss-C1_ss)/delta))
print 'The difference Ybar - Cbar - delta * Kbar is: ', rcdiff_ss
print(Y1_ss - C1_ss - delta * (K_ss-((Y2_ss-C2_ss)/delta)))

