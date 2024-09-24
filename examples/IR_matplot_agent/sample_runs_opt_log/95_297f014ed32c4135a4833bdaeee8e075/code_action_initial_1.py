import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
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

# Get the highest temperatures for 2015
temps_2015 = monthly_max_temps[monthly_max_temps['Year'] == 2015]

# Prepare the angles for the polar plot (12 months)
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Prepare the temperature values for 2015
temps_2015_values = [temps_2015.loc[temps_2015['Month'] == month, 'Temperature'].values[0] if month in temps_2015['Month'].values else np.nan for month in range(1, 13)]

# Add the first month to the end to close the circle
temps_2015_values += temps_2015_values[:1]  # Close the loop
angles += angles[:1]  # Close the loop

# Create polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot the circular points for the temperature data
ax.scatter(angles[:-1], temps_2015_values[:-1], color='red', s=100, label='Highest Temp (2015)', zorder=5)

# Plot the blue curve connecting the data points
ax.plot(angles, temps_2015_values, color='blue', linewidth=2, label='Temperature Curve (2015)', zorder=3)

# Set the title and labels
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom', fontsize=16)
ax.set_xticks(angles[:-1])  # Set the ticks for each month
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Month labels

# Add a legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Save the plot to a PNG file
plt.savefig("novice.png")

# Show the plot
plt.show()