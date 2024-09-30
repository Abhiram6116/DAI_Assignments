# Importing python libraries
import matplotlib.pyplot as plt
import pandas as pd

# sample data
data = {
    'Category': ['Start', 'Sales', 'Returns', 'Marketing', 'R&D', 'End'],
    'Amount': [1000, 300, -50, -200, -100, 950]
}
df = pd.DataFrame(data)

# Calculating the running totals
df['Running_Total'] = df['Amount'].cumsum()
df['Shifted_Total'] = df['Running_Total'].shift(1).fillna(0)
df['Position'] = df.apply(lambda row: row['Shifted_Total'] if row['Amount'] >= 0 else row['Running_Total'], axis=1)
# plotting the waterfall chart
fig, ax = plt.subplots(figsize=(10, 6))

# code for Bars
ax.bar(df['Category'], df['Amount'], bottom=df['Position'], color=['#4CAF50' if x >= 0 else '#F44336' for x in df['Amount']])

# code for lines to connect the bars
for i in range(1, len(df)):
    ax.plot([i-1, i], [df['Running_Total'][i-1], df['Running_Total'][i]], color='black')

# Adding the total labels
for i, (total, amount) in enumerate(zip(df['Running_Total'], df['Amount'])):
    ax.text(i, total + (amount / 2), f'{total:.0f}', ha='center', va='bottom' if amount > 0 else 'top')

ax.set_title('Waterfall Chart')
ax.set_ylabel('Amount')
plt.savefig('waterfall_plot.png')

