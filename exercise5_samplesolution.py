import numpy as np 
import matplotlib.pyplot as plt 

T = 50  # Time horizon
N = 10000  # Choose appropriate step size

def brownian_mot(N,T):  # Function to simulate a Brownian motion
    mu, sigma = 0, np.sqrt(1/N) # Mean and standard deviation 
    s = np.random.normal(mu, sigma, N*T) # N*T normal distributed random numbers 
    B = np.zeros(N*T+1)  # Vector creation
    for i in range(N*T):
        B[i+1]=B[i]+s[i] # Increments of BM on each time step
    return B   # Return the Brownian motion

def covariation(X,Y,N,T):  # Function to approximate the covariation of X and Y
    V = np.zeros(N*T+1)  # Vector creation
    for i in range(N*T):
        V[i+1]=V[i]+(X[i+1]-X[i])*(Y[i+1]-Y[i]) # Definition of the pth-Variation
    return V

# Part a)
B1 = brownian_mot(N,T)  # Two (independent) realizations of Brownian motions
B2 = brownian_mot(N,T)

rho = 0.5
B3 = rho*B1 + np.sqrt(1-rho**2)*B2 # Construction of \tilde{B}

x = np.linspace(0, T, N*T+1)   # X-axis discretization
plt.plot(x,covariation(B1,B2,N,T), label=r'$Cov(B_1,B_2)$')
plt.plot(x,covariation(B1,B3,N,T), label=r'$Cov(B_1,\tilde{B})$')
plt.plot(x,covariation(B2,B3,N,T), label=r'$Cov(B_2,\tilde{B})$')
plt.title("(i) Covariances, with " + r'$\rho=$' + str(rho) + r'$, \sqrt{1-\rho^2}\approx $' + str(np.sqrt(1-rho**2).round(2)))
plt.xlabel('t') # Labeling x-axis
plt.legend(loc='best') # Legend at the position where Python finds it the best
plt.show()

# Part b)

# Approximate process X with square-root diffusion coefficient
B = brownian_mot(N,T) # Brownian motion realization
X = np.zeros(N*T+1) 
for i in range(N*T):
    X[i+1] = X[i] + 1/N + np.sqrt(2*i/N)*(B[i+1]-B[i])  # Stochastic integral approximation scheme

# Approximate process X2 with exponential diffusion coefficient
X2 = np.zeros(N*T+1) 
for i in range(N*T):
    X2[i+1] = X2[i] + 1/N + 5/(np.sqrt(2))*(i/N)*(B[i+1]-B[i])  # Stochastic integral approximation scheme

x = np.linspace(0, T, N*T+1)   # X-axis discretization
plt.plot(x,covariation(X,X,N,T), label=r'$\langle X\rangle_t = t^2$')
plt.plot(x,covariation(X,X2,N,T), label=r'$\langle X,\tilde{X}\rangle_t = 2t^{\frac{5}{2}}$') 
plt.title("(ii) Quadratic (co-)variations of Ito processes")
plt.xlabel('t') # Labeling x-axis
plt.xlim(0,T) # Fix y-axis
plt.ylim(0,50*T) # Fix y-axis
plt.legend(loc='best') # Legend at the position where Python finds it the best
plt.show()