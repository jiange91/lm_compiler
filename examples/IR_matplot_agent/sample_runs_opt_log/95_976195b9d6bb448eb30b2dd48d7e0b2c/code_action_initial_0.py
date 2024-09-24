import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from CSV
data = pd.read_csv('data.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the specified date range
filtered_data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Extract month and year
filtered_data['Month'] = filtered_data['Date'].dt.month
filtered_data['Year'] = filtered_data['Date'].dt.year

# Get the highest temperature for each month
monthly_max_temps = filtered_data.groupby(['Month', 'Year'])['Temperature'].max().reset_index()

# Prepare data for polar plot
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Get the highest temperatures for each month in 2015
temps_2015 = monthly_max_temps[monthly_max_temps['Year'] == 2015]['Temperature'].tolist()

# Get the highest temperatures for all years for plotting
all_years_temps = monthly_max_temps.groupby('Month')['Temperature'].mean().tolist()

# Create polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot all years' average temperatures
ax.scatter(angles, all_years_temps, color='gray', alpha=0.5, label='Average Temperature (2004-2015)')

# Plot 2015 temperatures with circular points slightly offset
offset = 0.1  # Offset for circular points
ax.scatter(angles, temps_2015, color='red', label='Highest Temperature (2015)', zorder=5)

# Connect the 2015 data points with a blue curve
ax.plot(angles, temps_2015, color='blue', linewidth=2)

# Set the labels for each month
ax.set_xticks(angles)
ax.set_xticklabels(months)

# Set the title and legend
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()