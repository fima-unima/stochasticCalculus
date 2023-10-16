import numpy as np 
import matplotlib.pyplot as plt 

T = 5  # Time horizon
N = 10000  # Choose appropriate step size

def brownian_mot(N,T):  # Function to simulate a Brownian motion
    mu, sigma = 0, np.sqrt(1/N) # Mean and standard deviation 
    s = np.random.normal(mu, sigma, N*T) # N*T normal distributed random numbers 
    B = np.zeros(N*T+1)  # Vector creation
    for i in range(N*T):
        B[i+1]=B[i]+s[i] # Increments of BM on each time step
    return B   # Return the Brownian motion

B = brownian_mot(N,T) # Brownian motion realization
#(i)
Y = np.zeros(N*T+1)  # Vector for Euler solution
Y[0] = 1 # Starting at 1
Y_calculated = np.zeros(N*T+1) # vector for exact solution
Y_calculated[0] = 1
for i in range(N*T):
    Y[i+1] = Y[i] + Y[i]/N + Y[i]*(B[i+1]-B[i])  # Euler-Maruyama scheme
    Y_calculated[i+1] = np.exp(i/(2*N))*np.exp(B[i]) # Exact solution

x = np.linspace(0, T, N*T+1)   # X-axis discretization
plt.plot(x,Y, label="Euler-Maruyama scheme for (i)")
plt.plot(x,Y_calculated, label=r'$X_t=e^{\frac{1}{2}t+B_t}$' + " (geometric BM)")

#(ii)
Y2 = np.zeros(N*T+1)  # Vector for Euler solution
Y2[0] = 1 # Starting at 1
for i in range(N*T):
    Y2[i+1] = Y2[i] - Y2[i]/N + (B[i+1]-B[i])  # Euler-Maruyama scheme

plt.plot(x,Y2, label="Euler-Maruyama scheme for (ii)")
plt.title("Euler-Maruyama schemes")
plt.xlabel('t') # Labeling x-axis
plt.legend(loc='best') # Legend at the position where Python finds it the best
plt.ylim([ min(min(Y2),0) *1.1 , min(5*T, max(max(Y), max(Y2)))*1.1 ])  # Good scaling of the y-axis
plt.show()
