# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the Data
data = pd.read_csv('data.csv')

# Step 3: Data Preparation
# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the specified date range
filtered_data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Extract month and year
filtered_data['Month'] = filtered_data['Date'].dt.month
filtered_data['Year'] = filtered_data['Date'].dt.year

# Get the highest temperature for each month
monthly_max_temps = filtered_data.groupby(['Year', 'Month'])['Temperature'].max().reset_index()

# Step 4: Prepare Data for Polar Plot
# Create a list of month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Prepare data for the polar plot
monthly_avg_temps = monthly_max_temps.groupby('Month')['Temperature'].mean().reindex(range(1, 13))

# Get the temperatures for 2015
temps_2015 = monthly_max_temps[monthly_max_temps['Year'] == 2015]['Temperature'].values

# Ensure temps_2015 has 12 values (fill with NaN if necessary)
if len(temps_2015) < 12:
    temps_2015 = np.concatenate([temps_2015, [np.nan] * (12 - len(temps_2015))])

# Create angles for each month
angles = np.linspace(0, 2 * np.pi, len(month_labels), endpoint=False).tolist()

# Step 5: Create the Polar Plot
# Create a polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot the average temperatures
ax.scatter(angles, monthly_avg_temps, color='orange', s=100, label='Monthly Avg Temp', alpha=0.6)

# Plot the temperatures for 2015
# Offset the points slightly to prevent alignment
offset = 0.1
ax.scatter(angles, temps_2015 + offset, color='blue', s=100, label='2015 Temp', alpha=0.8)

# Connect the 2015 data points with a blue curve
ax.plot(angles, temps_2015 + offset, color='blue', linewidth=2)

# Set the labels for each month
ax.set_xticks(angles)
ax.set_xticklabels(month_labels)

# Set the title and legend
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()