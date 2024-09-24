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

# Get the highest temperatures for each month in 2015
temps_2015 = monthly_max_temps[monthly_max_temps['Year'] == 2015]

# Prepare the angles for the polar plot (12 months)
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Create a list of temperatures for each month (fill with NaN for missing months)
temps = [monthly_max_temps[(monthly_max_temps['Year'] == year) & (monthly_max_temps['Month'] == month)]['Temperature'].values[0] if not monthly_max_temps[(monthly_max_temps['Year'] == year) & (monthly_max_temps['Month'] == month)].empty else np.nan for month in range(1, 13)] for year in range(2004, 2016)]

# Create the polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot the data for each year
for i, year_data in enumerate(temps):
    # Offset the points slightly to prevent alignment
    offset = np.random.uniform(-0.1, 0.1, size=len(year_data))
    ax.scatter(angles, year_data + offset, label=str(2004 + i), alpha=0.5)

# Highlight the year 2015 with a blue curve
ax.plot(angles, temps_2015['Temperature'].values, color='blue', marker='o', label='2015', linewidth=2)

# Set the labels for the months
ax.set_xticks(angles)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Set the title and legend
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()