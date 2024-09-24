import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from CSV
data = pd.read_csv('data.csv')

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the specified date range
filtered_data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Extract month and year
filtered_data['Month'] = filtered_data['Date'].dt.month
filtered_data['Year'] = filtered_data['Date'].dt.year

# Get the highest temperature for each month
monthly_max_temps = filtered_data.groupby(['Year', 'Month'])['Temperature'].max().reset_index()

# Create a list of angles for each month (0 to 2π)
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Prepare data for the year 2015
temps_2015 = monthly_max_temps[monthly_max_temps['Year'] == 2015].set_index('Month')['Temperature']

# Create a full array for 2015 temperatures, filling missing months with NaN
temps_2015_full = [temps_2015.get(i, np.nan) for i in range(1, 13)]

# Prepare data for all years for plotting
all_years_temps = monthly_max_temps.groupby('Month')['Temperature'].max().values

# Create a polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot the data for all years
ax.scatter(angles, all_years_temps, color='gray', alpha=0.5, label='Monthly Max Temperatures (2004-2015)')

# Plot the data for 2015 with a blue curve
ax.plot(angles, temps_2015_full, color='blue', linewidth=2, label='2015 Temperatures')

# Add circular points for the 2015 data, slightly offset
offset = 0.1  # Offset for circular points
ax.scatter(angles, np.array(temps_2015_full) + offset, color='blue', s=100, zorder=5)

# Set the labels for each month
ax.set_xticks(angles)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Set the title and legend
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Save the plot to a PNG file
plt.savefig("novice.png")

# Show the plot
plt.show()