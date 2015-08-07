from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt

'''
Currently, the code converges for lower s-periods. It is not perfect, and 
I will keep trying to figure out exactly what is preventing it from converging
at higher S-period levels. It has to do with the current constraints in the 
f-solver.
'''


"""
This OLG Model calculates the time path and steady state for s periods, with 
labor endogenous, stationarizied population and GDP growth, and Taxes.

#### Paramaters ###
beta   = discount factor, (0,1)**years per period
delta   = depreciation, (0,1)**years per period
gamma    = consumption relative risk averstion, <1
sigma    = labor relative risk aversion, <1
alpha    = cobb-douglas ouput elasticity of labor (0,1)
A        = Firm productivity coeffecient >0
periods  = number of periods you wish to use (3,80)
years    = number of years per period (1,20)
error    = Intializes an error for the TPI >5
xi       = Used to calculate your convex combonation for TPI
epsilon  = Accuracy parameter in TPI calculation
T        = Number of periods to reach the steady state, should be large
shock    = How much to shock the economy away from the Steady State

guess_vector = intial guess for the cap fsolve. The first s entries are capital guess, the last s entries are labor guesses
show_graph = boolean variable, plots graph if True
"""

### Paramaters ###
periods = 20
years = 60./periods
beta = .96**years
delta = 1-(1-.05)**years
gamma = 2.9
alpha = .35
sigma = 2.9
A = 1
xi = .5
epsilon = 10e-8
T = 70
shock = .7
labor_guess = .8
cap_guess = .1
error = 1.
show_graph = True


guess_vector = np.ones(2*periods-1)*cap_guess
guess_vector[-periods:] = labor_guess

def wage (kvector, lvector):
    market_k = np.sum(kvector)
    market_l = np.sum(lvector)
    w = (1-alpha)*A*((market_k)/market_l)**(alpha)
    return w

def rate (kvector, lvector): 
    market_k = np.sum(kvector)
    market_l = np.sum(lvector)
    r = (alpha)*A*(market_l/(market_k))**(1-alpha)
    return r

def cap(guess_vector):
    """
    This takes the Euler equations, and sets them equal to zero for an f-solve
    Remember that Keq was found by taking the derivative of the sum of the 
        utility functions, with respect to k in each time period, and that 
        leq was the same, but because l only shows up in 1 period, it has a
        much smaller term.

    ### Paramaters ###
    guess_vector: The first half is the intial guess for the kapital, and
        the second half is the intial guess for the labor
    """
    #equations for keq
    ks = np.zeros(periods)
    ks[1:] = guess_vector[:periods-1]
    ls  = guess_vector[periods-1:]
    kk  = ks[:-1]
    kk1 = ks[1:]
    kk2 = np.zeros(periods-1)
    kk2[:-1] = ks[2:]
    lk  = ls[:-1]
    lk1 = ls[1:]
    #equation for leq
    ll = np.copy(ls)
    kl = np.copy(ks)
    kl1 = np.zeros(periods)
    kl1[:-1] = kl[1:]
    w = wage(ks, ls)
    r = rate(ks, ls)
    keq = ((lk*w+(1.+r-delta)*kk - kk1)**-gamma - (beta*(1+r-delta)*(lk1*w+(1+r-delta)*kk1-kk2)**-gamma))
    leq = ((w*(ll*w + (1+r-delta)*kl-kl1)**-gamma)-(1-ll)**-sigma)
    error = np.append(keq, leq)

    return np.append(keq, leq)

ssvalue = optimize.fsolve(cap, guess_vector)
kbars = np.zeros(periods -1)
kbars = ssvalue[:periods-1]
lbars = np.zeros(periods)
lbars = ssvalue[periods-1:]
print 'Capital steady state values: {}'.format(kbars)
print 'Labor steady state values: {}'.format(lbars)
wbar =  wage(kbars, lbars)
rbar = rate(kbars, lbars)
print("Wage steady state: ", wbar)
print("Rate stead state: ", rbar)
Kss = np.sum(kbars)
Lss = np.sum(lbars)
print('Market Capital steady state: ', Kss)
print('Market Labor steady state: ', Lss)
K0 = Kss*shock
kshock = kbars*shock

#################### Exercises 3,4 ##############################

def wage_path(K_guess,L_guess):
    """
    Creates and returns the wage path vector
    Paramaters
    K_guess  = array of n capital guesses
    L_guess  = array of n labor guesses
    """
    return list((1-alpha)*(K_guess/L_guess)**alpha)

def rate_path(K_guess, L_guess):
    """
    Creates and returns the rate path vector
    Paramaters
    K_guess  = array of n capital guesses
    L_guess  = array of n labor guesses
    """
    path = list(alpha*(L_guess/K_guess)**(1-alpha))
    return path

def L2_norm_func(path1, path2):
    """
    Measures the distance between the two calculated TPIs, 
    returns the L2 norm between them.
    Parameters
    path1 = array of S elements
    path2 = array of S elements
    """
    dif = path1 - path2
    sq = dif ** 2
    summed = sq.sum()
    rooted = summed ** .5
    return rooted

def TPI_Euler(guess, wage, rate, kbars, counter):
    '''
    This is a period general version of the equations, that will take in vectors
    for each of the values, and return the same number of equations to optimize. 
    K_guess will be the capital values thus calculated,
    wage will be the wage vector that we can pull wage1, wage2 from
    rate will be the rate vector that we can pull rate1 and rate2 from
    returns an array of S-1 Euler errors to be minimized by an fsolve
    '''

    kbars = np.append(np.array([0.]),kbars)
    k_guess = guess[:counter-1]
    l_guess = guess[counter-1:]
    wage1 = wage[:counter-1]
    wage2 = wage[1:counter]
    rate1 = rate[:counter-1]
    rate2 = rate[1:counter]
    k1 = np.zeros(counter-1)
    k1[0] = kbars[-counter]
    k2 = np.copy(k_guess)
    k3 = np.zeros(counter-1)
    if counter >2: 
        k1[1:] = k_guess[:-1]
        k3[:-1] = k_guess[1:]
    l1 = l_guess[:-1]
    l2 = l_guess[1:]
    w = wage[:counter]
    l = np.copy(l_guess)
    r = rate[:counter]
    lk1 = np.append(k1, k_guess[-1])
    lk2 = np.append(k2, 0)
     



    error1 = ((l1*wage1 + (1+rate1 - delta)*k1-k2)**-gamma - beta * (1+rate2 - delta) * 
            (wage2*l2 + (1+rate2 - delta)*k2 - k3)**-gamma)
    error2 = w*(l*w + (1+r-delta)*lk1-lk2)**-gamma-(1-l)**-gamma

    mask2 = l_guess < 0
    error2[mask2] += 1e12


    return np.append(error1, error2)

def TPI_Euler_2(guess, wage, rate, K_guess_init, counter):
    '''
    This is a period general version of the equations, that will take in vectors
    for each of the values, and return the same number of equations to optimize. 
    K_guess will be the capital values thus calculated,
    wage will be the wage vector that we can pull wage1, wage2 from
    rate will be the rate vector that we can pull rate1 and rate2 from
    '''

    k_guess = guess[:periods-1] 
    l_guess = guess[periods-1:]
    wagess = np.ones(periods + T + 2) * wage[-1]
    ratess = np.ones(periods + T + 2) * rate[-1]
    wagess[:T] = wage
    ratess[:T] = rate
    wage = wagess
    rate = ratess
    wage1 = np.ones(periods-1) * wage[-1]
    wage2 = np.ones(periods-1) * wage[-1]
    wage1 = wage[counter:periods + counter-1]
    wage2 = wage[1+counter:periods+counter]
    rate1 = np.ones(periods-1) * rate[-1]
    rate2 = np.ones(periods-1) * rate[-1]
    rate1 = rate[counter:periods-1 + counter]
    rate2 = rate[1+counter:periods+counter]
    k1 = np.zeros(periods-1)
    k1[1:] = k_guess[:-1]
    k2 = np.copy(k_guess)
    k3 = np.zeros(periods-1)
    k3[:-1] = k_guess[1:]
    l1 = l_guess[:-1]
    l2 = l_guess[1:]
    w = wage[counter+1:counter+1+periods]
    l = np.copy(l_guess)
    r = rate[counter+1:counter+1+periods]
    lk1 = np.append(k1, k_guess[-1])
    lk2 = np.append(k2, 0)

    error1 = ((l1*wage1 + (1+rate1 - delta)*k1-k2)**-gamma - beta * (1+rate2 - delta) * 
            (wage2*l2 + (1+rate2 - delta)*k2 - k3)**-gamma)
    error2 = w*(l*w + (1+r-delta)*lk1-lk2)**-gamma-(1-l)**-gamma

    #mask1 = l_guess < 0
    #error2[mask1] += 1e12
    

    return np.append(error1, error2)

def Scaler_Euler(labor, wage, rate, kap):
    return wage*(labor*wage + (1+rate-delta)*kap)**-gamma-(1-labor)**-gamma

print 'Working on TPI...'
K_new = np.linspace(K0, Kss, T)
L_new = np.ones(T)*Lss
iters = 0
while error > epsilon:
    iters += 1
    counter = 2
    K_old = np.copy(K_new)
    L_old = np.copy(L_new)
    wage_guess = np.asarray(wage_path(K_new,L_new))
    rate_guess = np.asarray(rate_path(K_new,L_new))
    K_matrix = np.zeros((T+periods,periods-1))
    K_matrix[0,:] = kshock
    L_matrix = np.zeros((T+periods,periods))
    lcorner = optimize.fsolve(Scaler_Euler, lbars[-1], args =(wage_guess[0], rate_guess[0], kbars[-1]))
    L_matrix[0,-1] = lcorner
    while counter <= periods:
        guess = np.ones(counter*2-1)
        guess[:counter-1] = kbars[periods-counter:]
        guess[counter-1:] = lbars[periods-counter:]
        newvec = optimize.fsolve(TPI_Euler, guess, args = (wage_guess, rate_guess, kshock, counter))
        newk = newvec[:counter-1]
        newl = newvec[counter-1:]
        if counter == periods:
            K_matrix[:periods,:] += np.diag(newk, -1)[:,:-1]
        else:
            K_matrix[1:periods-1, 1:] += np.diag(newk, periods-1-counter)
        L_matrix[:periods,:] += np.diag(newl, periods-counter)
        counter +=1

    for t_period in xrange(T):
        guess = np.ones(periods*2-1)
        guess[:periods-1] = kbars
        guess[periods-1:] = lbars
        newvec = optimize.fsolve(TPI_Euler_2, guess, args = (wage_guess, rate_guess, K_new, t_period))
        newk = newvec[:periods-1]
        newl = newvec[periods-1:]
        K_matrix[t_period+2:periods+t_period+1, :periods] += np.diag(newk, 0)
        L_matrix[t_period+1:periods+t_period+1, :periods]+= np.diag(newl, 0)
        t_period += 1

    K_new = K_matrix.sum(axis = 1)
    K_new = K_new[:T]
    L_new = L_matrix.sum(axis = 1)
    L_new = L_new[:T]
    Kerror = L2_norm_func(K_new, K_old)
    Lerror = L2_norm_func(L_new, L_old)
    error = max(Kerror, Lerror)

    print error
    if error > epsilon:
        K_new = xi * K_new + (1-xi) * K_old
        L_new = xi * L_new + (1-xi) * L_old



### Plot this sucker! ###
x = np.linspace(0,T,T)+1
fig = plt.figure()
fig.suptitle("TPI for Capital and Labor", fontsize = 16)

ax = plt.subplot("211")
ax.set_title("Capital TPI")
ax.plot(x,K_new)

ax = plt.subplot("212")
ax.set_title("Labor TPI")
ax.plot(x, L_new)
if show_graph == True:
    plt.show()

