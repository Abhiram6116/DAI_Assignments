# Uses of Violin Plot
# A violin plot is a powerful data visualization tool with several key uses:

# 1. Visualizing Distribution
# Combines box plot and kernel density for smooth data density representation.
# Facilitates comparison of distributions across multiple categories.
# 2. Identifying Modalities
# Reveals multiple modes (peaks) in the data, indicating subpopulations.
# 3. Understanding Variability
# Illustrates spread and variability of data in more detail than box plots.
# 4. Highlighting Outliers
# Helps in the detection of outliers within the data distribution.
# 5. Aesthetic Appeal
# Offers a visually appealing way to present data.
# Conclusion
# Violin plots are effective for comparing distributions, identifying patterns, and providing insights into data variability.

import seaborn
import pandas as pd
import matplotlib.pyplot as plt

seaborn.set(style = "whitegrid")
sampledata = pd.read_csv("Test set.csv")

plt.figure(figsize=(10,8))
seaborn.violinplot(x ="Sex",y="Weight",data=sampledata)

plt.savefig('violin_plot.png')