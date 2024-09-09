import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Data
data = pd.read_csv('data.csv')

# Data Preparation
data['Date'] = pd.to_datetime(data['Date'])
filtered_data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]
filtered_data['Month'] = filtered_data['Date'].dt.month
filtered_data['Year'] = filtered_data['Date'].dt.year

# Get the highest temperature for the first of each month
monthly_max = filtered_data.groupby(['Year', 'Month'])['Temperature'].max().reset_index()

# Extract data for 2015
data_2015 = monthly_max[monthly_max['Year'] == 2015]

# Prepare data for all years
all_years = monthly_max['Temperature'].values
months = monthly_max['Month'].values

# Create Polar Plot
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)  # 12 months
radii = np.zeros(12)

# Fill the radii with the maximum temperatures
for month in range(1, 13):
    month_data = monthly_max[monthly_max['Month'] == month]
    if not month_data.empty:
        radii[month - 1] = month_data['Temperature'].max()

# Add a small offset to the angles to prevent alignment
offset = 0.1
theta += offset

# Plot the Data
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})

# Plot all years
ax.scatter(theta, radii, color='orange', label='Monthly Max Temp (2004-2015)', s=100)

# Plot 2015 data
theta_2015 = np.linspace(0, 2 * np.pi, len(data_2015), endpoint=False) + offset
ax.plot(theta_2015, data_2015['Temperature'], color='blue', marker='o', label='2015 Data', linewidth=2)

# Set the labels for each month
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(theta)  # Set the ticks to the month positions
ax.set_xticklabels(month_labels)

# Add title and legend
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')
ax.legend(loc='upper right')

# Save the plot to a file
plt.savefig('plot.png')

# Show the plot
plt.show()