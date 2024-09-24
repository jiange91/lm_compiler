import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Data
data = pd.read_csv('data.csv')

# Data Preparation
# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the years 2004 to 2015
filtered_data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Extract month and year
filtered_data['Month'] = filtered_data['Date'].dt.month
filtered_data['Year'] = filtered_data['Date'].dt.year

# Get the highest temperature for the first of each month
monthly_max = filtered_data.groupby(['Year', 'Month'])['Temperature'].max().reset_index()

# Prepare Data for Polar Plot
# Create a list of angles for each month (0 to 2π)
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)

# Prepare temperature data for plotting
temperatures = monthly_max[monthly_max['Year'] == 2015]['Temperature'].values
temperatures = np.concatenate((temperatures, [temperatures[0]]))  # Close the loop for the plot

# Create the Polar Plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})

# Plot the temperatures
ax.plot(angles, temperatures, color='blue', marker='o', label='2015 Temperatures')

# Add circular points slightly offset
offset = 0.1  # Offset for circular points
for i in range(len(temperatures) - 1):
    ax.plot([angles[i], angles[i]], [temperatures[i], temperatures[i] + offset], color='blue')

# Set the labels for each month
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(angles[:-1])  # Set the ticks for each month
ax.set_xticklabels(month_labels)

# Add title and legend
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')
ax.legend(loc='upper right')

# Save the plot to a PNG file
plt.savefig("novice.png")

# Show the plot
plt.show()