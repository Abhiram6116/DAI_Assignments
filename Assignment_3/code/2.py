import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#***************************************************************# 
# 2.1 

# Custom Epanechnikov KDE class
class EpanechnikovKDE:
    # The constructor of the class
    def __init__(self, bandwidth=1.0) :
        self.bandwidth = bandwidth
        self.data = None
    
    # fit function to store data ie update data member variable
    def fit(self, data) :
        self.data = np.array(data)
    
    # Implement the Epanechnikov kernel function
    def epanechnikov_kernel(self, x, xi) :
        # we are implementing K((x-xi)/h).
        # Calculate the Euclidean distance ||x||2
        distance = np.linalg.norm(x - xi)
        # Compute K((x - xi) / h)
        if distance <= self.bandwidth: # |x|<=1
            return (2/np.pi) * (1 - (distance**2)/(self.bandwidth**2))
        else:
            return 0  # |x|>1
    
    def evaluate(self, x) :
        # Function call will be expensive, Lets just use 
        # np array properties for efficient computation.
        distances = np.linalg.norm(x - self.data, axis=1)
        # Use where to calculate kernel values
        kernel_values = np.where(distances <= self.bandwidth, (2/np.pi) * (1 - (distances**2)/(self.bandwidth**2)), 0)
        density = np.sum(kernel_values) / ((self.bandwidth**2) * len(self.data))
        
        # return density
        return density

#***************************************************************#
# 2.2
# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']

# STEP 1 : Instantiate the class with appropriate band width
# Band Width is h
h = 0.25
kde = EpanechnikovKDE(bandwidth=h) 

# STEP 2 : Fit the data
kde.fit(data) 

# STEP 3: Estimate and plot the probability density of transactions
x_range = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
y_range = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)

X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

# Evaluate the density for each point in the meshgrid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = kde.evaluate(np.array([X[i, j], Y[i, j]]))

# STEP 4: Plotting in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_title(f'Probability Density Estimate of Kernel Estimate using bandwidth = {h}')
ax.set_xlabel('Transaction Amount X')
ax.set_ylabel('Transaction Amount Y')
ax.set_zlabel('Probability Density')

plt.savefig('images/transaction_distribution.png')
plt.clf()

# Count number of modes.
# Lets call a node mode if it is greater than its immediate 8 neighbours
# Ignore boundaries as we can deduce that no maximas are there at boundaries

mode_count = 0
row_op = [0, 0, 1, 1, 1, -1, -1, -1]
col_op = [1, -1, 0, 1, -1, 0, 1, -1]
rows, cols = Z.shape
for i in range(1, rows-1):
    for j in range(1, cols-1):
        curr_val = Z[i, j]
        is_max = True
        for k in range(8):
            if(Z[i+row_op[k], j+col_op[k]]>curr_val):
                is_max = False
                break
        if(is_max) : mode_count+=1

print("Number of modes = ", mode_count)



