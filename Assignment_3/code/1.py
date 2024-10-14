import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the CSV file data1.csv
data = pd.read_csv('data1.csv')

# Filter the first 1500 rows (1st step of filtering)
data_filtered = data.head(1500)

# Second filter that is column D (Mpc) should be less than 4
data_filtered = data_filtered[data_filtered['D (Mpc)'] < 4]

#*********************************************************************#
# Part A

#  We need to plot a histogram with 10 bins
# We are plotting the normalized one ie probability density
plt.hist(data_filtered['D (Mpc)'], bins=10, edgecolor='black', density=True)

# Set labels and title
plt.xlabel('Distance (Mpc)')
plt.ylabel('Probability density')
plt.title('Histogram of Metric Distance (10 bins)')

# Save the plot as '10binhistogram.png'
plt.savefig('images/10binhistogram.png')
plt.clf()

# Now calculate individual probabilities
# Calculate the bin counts and edges 
counts, bin_edges = np.histogram(data_filtered['D (Mpc)'], bins=10)

# The width of the data we are using
data_width = bin_edges[-1] - bin_edges[0]

# Calculate probabilities by normalizing counts
n = len(data_filtered['D (Mpc)'])  # Total number of data points = n
probabilities = counts / n  # Estimated probabilities pj

# Print the estimated probabilities for each bin
print("Estimated probabilities for each bin (p̂j): ")
for i in range(len(probabilities)):
    print(f"Bin {i+1}: {probabilities[i]: .4f}")

# Print bin edges for reference
print("\nBin edges (Mpc):", bin_edges)

# done

#**********************************************************************#
# Part B

# The histogram is underfit as it has been oversmoothed due to too few bins,
# which results in a loss of important details. This underfitting prevents the histogram from 
# accurately representing the underlying distribution of the data ie has higher loss function
# We need to increase the number of bins
# so as to find the optimal bin width ie that which minimizes loss

# done

#******************************************************************#
# Part C

#  we have cross validation score => J(h) = 2/(n-1)h  - (n+1)/(n-1)h ∑(pj^2)
# We need to plot the score vs Number of bins m for m = 1 to 1000

m_values = range(1, 1001) 
J_values = np.zeros(1000) 

for m in m_values:
    # We need to calculate the scores.
    counts, bin_edges = np.histogram(data_filtered['D (Mpc)'], bins=m)
    probabilities = counts/n
    
    h = data_width/m
    J_values[m-1] = 2/((n-1)*h) - (n+1)*np.sum(probabilities**2)/((n-1)*h)

# Now time for the plot
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(m_values, J_values, label='Cross-Validation Score J(h)', color='blue')
plt.title('Cross-Validation Score vs Number of Bins')
plt.xlabel('Number of Bins (m)')
plt.ylabel('Cross-Validation Score J(h)')
plt.grid()
plt.legend()
plt.savefig('images/crossvalidation.png')
plt.clf()

# done

#*******************************************************************#
# Part D

#  Find optimal bin_width that is the one that minimizes cross validation score
min_index = np.argmin(J_values)
print("\nNumber of bins for optimal h = ", min_index+1)
min_score_width = data_width/(min_index+1)
print("Optimal h = ", min_score_width) 


#******************************************************************#
# Part E

#  Let us plot the histogram with the optimal bin width
plt.hist(data_filtered['D (Mpc)'], bins=(min_index+1), edgecolor='black', density=True)
# Set labels and title
plt.xlabel('Distance (Mpc)')
plt.ylabel('Probability density using optimal $h^*$')
plt.title('Histogram of Metric Distance with optimal $h^*$')
# Save the plot as optimalhistogram.png
plt.savefig('images/optimalhistogram.png')

# **Detail and Information Capture**: The histogram with m = 50 captures significantly
# more detail than the one with  m = 10 . 
# While the histogram with  m = 10  appears smooth and oversimplified, 
# the m = 50  histogram reveals additional features in the data distribution.

# **Peaks and Minima**: The histogram with m = 50  shows new peaks and minima 
# that are not visible in the  m = 10  histogram. 
# This suggests a more complex structure in the data, 
# indicating the presence of clusters or gaps that might be 
# important for understanding the distribution of distances.

# Done

#*******************************************************************#
# Part F 
# Done

