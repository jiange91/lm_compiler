import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Data preparation
data['Date'] = pd.to_datetime(data['Date'])
mask = (data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')
filtered_data = data.loc[mask]
filtered_data['Month'] = filtered_data['Date'].dt.month
filtered_data['Year'] = filtered_data['Date'].dt.year
monthly_max = filtered_data.groupby(['Year', 'Month'])['Temperature'].max().reset_index()

# Extract data for plotting
data_2015 = monthly_max[monthly_max['Year'] == 2015]

# Create polar plot
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
radii = np.zeros(12)
for month in range(1, 13):
    max_temp = monthly_max[monthly_max['Month'] == month]['Temperature'].max()
    radii[month - 1] = max_temp

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

for i in range(len(radii)):
    ax.plot([theta[i], theta[i]], [0, radii[i]], color='gray', alpha=0.5)
    ax.scatter(theta[i], radii[i], s=100, label=f'{i+1} Month', edgecolor='black', alpha=0.7)

ax.plot(theta, data_2015['Temperature'], color='blue', linewidth=2, label='2015 Temperatures')

# Add labels and title
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(theta)
ax.set_xticklabels(month_labels)
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

# Save the plot
plt.tight_layout()
plt.savefig('plot.png')
plt.show()