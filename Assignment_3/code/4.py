import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('glass_data.txt', sep='\s+')  # Adjust based on your file
x = data['Al'].values  
y = data['RI'].values

# Kernel functions
def gaussian_kernel(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u ** 2)

def epanechnikov_kernel(u):
    return 0.75 * (1 - u ** 2) * (np.abs(u) <= 1)

# Nadaraya-Watson kernel regression implementation
def nadaraya_watson_kernel_regression(x_train, y_train, x_test, h, kernel):
    y_pred = np.zeros_like(x_test)
    for i, x_t in enumerate(x_test):
        weights = kernel((x_t - x_train) / h)
        if np.sum(weights) != 0:
            y_pred[i] = np.sum(weights * y_train) / np.sum(weights)
        else:
            y_pred[i] = np.mean(y_train)
    return y_pred

# Leave-One-Out Cross-Validation (LOOCV) for bandwidth selection
def loocv(x, y, h_values, kernel):
    n = len(x)
    risks = []

    for h in h_values:
        total_risk = 0
        for i in range(n):
            # Leave one out
            x_train = np.delete(x, i)
            y_train = np.delete(y, i)
            x_test = np.array([x[i]])
            y_test = np.array(y[i])

            # Perform kernel regression
            y_pred = nadaraya_watson_kernel_regression(x_train, y_train, x_test, h, kernel)
            total_risk += (y_test - y_pred) ** 2  
        
        risks.append(total_risk / n)

    return risks

# Bandwidth values to test
h_values = np.linspace(0.05, 0.5, 50)

# Perform LOOCV to estimate risks for Gaussian kernel
risks_gaussian = loocv(x, y, h_values, gaussian_kernel)

# Find the optimal bandwidth for Gaussian kernel
min_risk_index_gaussian = np.argmin(risks_gaussian)
optimal_h_gaussian = h_values[min_risk_index_gaussian]
print(f'Gaussian Optimal bandwidth: {optimal_h_gaussian}, Minimum estimated risk: {risks_gaussian[min_risk_index_gaussian]}')

# Perform LOOCV to estimate risks for Epanechnikov kernel
risks_epanechnikov = loocv(x, y, h_values, epanechnikov_kernel)

# Find the optimal bandwidth for Epanechnikov kernel
min_risk_index_epanechnikov = np.argmin(risks_epanechnikov)
optimal_h_epanechnikov = h_values[min_risk_index_epanechnikov]
print(f'Epanechnikov Optimal bandwidth: {optimal_h_epanechnikov}, Minimum estimated risk: {risks_epanechnikov[min_risk_index_epanechnikov]}')

# Generate 100 equally spaced x values from min to max for smooth plotting
x_test = np.linspace(np.min(x), np.max(x), 100)

# Plotting results for Gaussian Kernel
plt.figure(figsize=(10, 8))

# Oversmoothed
y_oversmooth_gaussian = nadaraya_watson_kernel_regression(x, y, x_test, optimal_h_gaussian * 4, gaussian_kernel)
plt.subplot(2, 2, 1)
plt.scatter(x, y, color='blue',s = 6 ,label='Data')
plt.plot(x_test, y_oversmooth_gaussian, color='red', label='Oversmoothed (h=optimal_h*4)')
plt.title('Oversmoothed (Gaussian)')
plt.legend()

# Undersmoothed
y_undersmooth_gaussian = nadaraya_watson_kernel_regression(x, y, x_test, optimal_h_gaussian / 4, gaussian_kernel)
plt.subplot(2, 2, 2)
plt.scatter(x, y, color='blue',s=6, label='Data')
plt.plot(x_test, y_undersmooth_gaussian, color='green', label='Undersmoothed (h=optimal_h/4)')
plt.title('Undersmoothed (Gaussian)')
plt.legend()

# Just right
y_just_right_gaussian = nadaraya_watson_kernel_regression(x, y, x_test, optimal_h_gaussian, gaussian_kernel)
plt.subplot(2, 2, 3)
plt.scatter(x, y, color='blue',s=6, label='Data')
plt.plot(x_test, y_just_right_gaussian, color='orange', label='Just Right (h=optimal_h)')
plt.title('Just Right (Gaussian)')
plt.legend()

# Cross-validation curve for Gaussian kernel
plt.subplot(2, 2, 4)
plt.plot(h_values, risks_gaussian, marker='o', markersize = 3)
plt.title('Cross-Validation Curve (Gaussian)')
plt.xlabel('Bandwidth (h)')
plt.ylabel('Estimated Risk')
plt.axvline(x=optimal_h_gaussian, color='red', linestyle='--', label='Optimal h')
plt.legend()

# Save and show plots
plt.tight_layout()
plt.savefig('images/gaussian_kernel_regression.png')


# Plotting results for Epanechnikov Kernel
plt.figure(figsize=(10, 8))

# Oversmoothed
y_oversmooth_epanechnikov = nadaraya_watson_kernel_regression(x, y, x_test, optimal_h_epanechnikov * 4, epanechnikov_kernel)
plt.subplot(2, 2, 1)
plt.scatter(x, y, color='blue', s = 6, label='Data')
plt.plot(x_test, y_oversmooth_epanechnikov, color='red', label='Oversmoothed (h=optimal_h*4)')
plt.title('Oversmoothed (Epanechnikov)')
plt.legend()

# Undersmoothed
y_undersmooth_epanechnikov = nadaraya_watson_kernel_regression(x, y, x_test, optimal_h_epanechnikov / 4, epanechnikov_kernel)
plt.subplot(2, 2, 2)
plt.scatter(x, y, color='blue', s = 6, label='Data')
plt.plot(x_test, y_undersmooth_epanechnikov, color='green', label='Undersmoothed (h=optimal_h/4)')
plt.title('Undersmoothed (Epanechnikov)')
plt.legend()

# Just right
y_just_right_epanechnikov = nadaraya_watson_kernel_regression(x, y, x_test, optimal_h_epanechnikov, epanechnikov_kernel)
plt.subplot(2, 2, 3)
plt.scatter(x, y, color='blue', s = 6, label='Data')
plt.plot(x_test, y_just_right_epanechnikov, color='orange', label='Just Right (h=optimal_h)')
plt.title('Just Right (Epanechnikov)')
plt.legend()

# Cross-validation curve for Epanechnikov kernel
plt.subplot(2, 2, 4)
plt.plot(h_values, risks_epanechnikov, marker='o', markersize = 3)
plt.title('Cross-Validation Curve (Epanechnikov)')
plt.xlabel('Bandwidth (h)')
plt.ylabel('Estimated Risk')
plt.axvline(x=optimal_h_epanechnikov, color='red', linestyle='--', label='Optimal h')
plt.legend()

# Save and show plots for Epanechnikov
plt.tight_layout()
plt.savefig('images/epanechnikov_kernel_regression.png')

# done
