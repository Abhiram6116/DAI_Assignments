# Optional: Plot detected peaks on the surface
# plt.figure(figsize=(12, 8))
# plt.contourf(X, Y, Z, levels=50, cmap='viridis')  # Plot KDE surface
# plt.colorbar(label='Density')

# # Mark detected peaks on the plot
# if peaks.size > 0:
#     plt.scatter(peaks[:, 0], peaks[:, 1], color='red', label='Peaks', marker='x')

# plt.title(f'2D KDE with Detected Peaks (Number of Peaks = {len(peaks)})')
# plt.xlabel('Transaction Amount X')
# plt.ylabel('Transaction Amount Y')
# plt.legend()
# plt.show()