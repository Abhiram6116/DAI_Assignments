# ## **Uses of Pareto Chart**

# A **Pareto chart** is a bar graph that helps identify significant factors in data analysis. Here are its key uses:

# ### **1. Problem Identification**
# - Highlights major issues.
# - Focuses on critical factors.

# ### **2. Decision Making**
# - Prioritizes actions based on impact.
# - Aids in resource allocation.

# ### **3. Quality Control**
# - Identifies and categorizes defects.
# - Tracks improvement effectiveness.

# ### **4. Performance Analysis**
# - Provides clear visual data representation.
# - Aids in recognizing trends.

# ### **5. Reporting**
# - Enhances communication of findings.
# - Supports arguments with visual data.

# ### **6. Continuous Improvement**
# - Drives initiatives in Lean and Six Sigma.
# - Facilitates ongoing evaluation.

# ### **7. Strategic Planning**
# - Identifies growth areas.
# - Supports long-term planning.

# Using a **Pareto chart** enables organizations to make informed, data-driven decisions.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

df = pd.DataFrame({'country': [3708,4489,225,290,410,90,164,170,30]})
df.index = ['United States','Russia','United Kingdom','France','China','Israel','India','Pakistan','North Korea']
df = df.sort_values(by='country',ascending=False)
df["cumpercentage"] = df["country"].cumsum()/df["country"].sum()*100

fig, ax = plt.subplots()
ax.bar(df.index, df["country"], color="C0")
ax2 = ax.twinx()
ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())

ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
plt.savefig('pareto_chart.png')