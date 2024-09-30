import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N = 100000 # Number of samples.

def sample(loc, scale):
    # Sample 100000 numbers from a uniform distribution.
    uniform_sample = np.random.uniform(0, 1, size=N)
    # Use the inverse CDF (ppf) to transform uniform samples to Gaussian samples.
    gaussian_sample = norm.ppf(uniform_sample, loc = loc, scale = scale)
    
    return gaussian_sample


# Parameters for the four Gaussian distributions
means = [0, 0, 0, -2]
variances = [0.2, 1.0, 5.0, 0.5]
stds = [(np.sqrt(variance)).item() for variance in variances]
colors = ['b', 'r', 'y', 'g']
labels = [f'$\\mu={mu}, \\sigma^{2}={variance}$' for mu, variance in zip(means, variances)]

# Create a figure with a single axis.
plt.figure(figsize = (12, 8))

# Plot samples for each parameter choice.
for mu, sigma, color, label in zip(means, stds, colors, labels):
    sample_gen = sample(mu, sigma)
    plt.hist(sample_gen, bins=500, density=True, alpha=0.6, color=color, label=label)

plt.legend()
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Samples from Gaussian Distributions')

plt.xlim(-6, 6) # To increase visibility in only the middle region

# Save the figure
plt.savefig('images/2c.png')
plt.show()

    
    
