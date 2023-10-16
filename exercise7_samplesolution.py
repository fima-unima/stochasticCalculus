import numpy as np 
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean 
import scipy.integrate as integrate 
import scipy.stats as stats

T = 100  # Time horizon
N = 10**6  # Sample size for Monte Carlo
S0 = 5 # Price of the risky asset at T=0
s = 1 # Volatility of the risky asset
K = 4 # Strike price of the call option

def payoff(x): # Payoff function of a european call option with strike K
    return np.maximum(x-K,np.zeros_like(x)) # x might also be a vector

def integrand(x): # Integrand of the price formula
    return stats.norm.pdf(x, S0, s*np.sqrt(T)) * payoff(x)

# (i)
price = integrate.quad(integrand,-np.inf,np.inf) # Valuation of the integral

#(ii)
price_mc = np.mean(payoff(np.random.normal(S0, s*np.sqrt(T), N)))  # Monte Carlo price is the mean over N evaluations of the payoff function on normally distributed random numbers

print("The (nearly) exact Bachelier-price is: " + str(price[0])) 
print("The Monte-Carlo-price is: " + str(price_mc))