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

# Prepare the angles for the polar plot (12 months)
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Create a list of temperatures for each month (fill with NaN for missing months)
monthly_temps = [monthly_max_temps[monthly_max_temps['Month'] == i]['Temperature'].max() if not monthly_max_temps[monthly_max_temps['Month'] == i].empty else np.nan for i in range(1, 13)]

# Create the polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot the temperatures for all years
ax.scatter(angles, monthly_temps, color='orange', label='Monthly Max Temp', s=100)

# Offset the points slightly to prevent alignment
offset = 0.1
ax.scatter(np.array(angles) + offset, monthly_temps, color='orange', s=100)

# Get the highest temperatures for 2015
temps_2015 = monthly_max_temps[monthly_max_temps['Year'] == 2015]['Temperature'].tolist()

# Ensure temps_2015 has 12 values for plotting
temps_2015_full = [temps_2015[i] if i < len(temps_2015) else np.nan for i in range(12)]

# Plot the temperatures for 2015 with a blue curve
ax.plot(angles, temps_2015_full, color='blue', linewidth=2, label='2015 Max Temp')

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