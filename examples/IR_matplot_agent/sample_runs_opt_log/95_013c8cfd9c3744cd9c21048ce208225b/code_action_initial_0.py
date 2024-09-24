import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data for the specified date range
filtered_data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Extract month and year
filtered_data['Month'] = filtered_data['Date'].dt.month
filtered_data['Year'] = filtered_data['Date'].dt.year

# Get the highest temperature for each month
monthly_max_temps = filtered_data.groupby(['Year', 'Month'])['Temperature'].max().reset_index()

# Create a list of month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Extract the highest temperatures for 2015
temps_2015 = monthly_max_temps[monthly_max_temps['Year'] == 2015]['Temperature'].values

# Prepare the angles for the polar plot (12 sectors)
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Repeat the first value to close the circular graph
temps_2015 = np.concatenate((temps_2015, [temps_2015[0]]))
angles += angles[:1]  # Close the circle

# Create the polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot the circular points for the temperature data
ax.scatter(angles[:-1], temps_2015[:-1], color='blue', s=100, label='2015 Max Temp', zorder=5)

# Connect the data points with a blue curve
ax.plot(angles, temps_2015, color='blue', linewidth=2, zorder=3)

# Set the labels for each month
ax.set_xticks(angles[:-1])
ax.set_xticklabels(month_labels)

# Set the title and legend
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()