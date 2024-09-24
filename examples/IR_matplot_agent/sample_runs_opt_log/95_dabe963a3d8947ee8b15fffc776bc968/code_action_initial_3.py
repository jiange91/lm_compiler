import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the years 2004 to 2015 and the first of each month
data = data[(data['Date'].dt.day == 1) & (data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Extract the maximum temperature for each month
monthly_data = data.groupby([data['Date'].dt.year, data['Date'].dt.month])['Temperature'].max().reset_index()
monthly_data.columns = ['Year', 'Month', 'Temperature']  # Rename columns for clarity

# Create a list of angles for each month
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Extract temperatures for the year 2015
temp_2015 = monthly_data[monthly_data['Year'] == 2015]['Temperature'].tolist()

# Prepare the temperatures for all months
monthly_temps = []
for month in range(1, 13):
    temps = monthly_data[monthly_data['Month'] == month]['Temperature'].tolist()
    monthly_temps.append(temps[0] if temps else np.nan)  # Use np.nan if no data for that month

# Create polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot the temperatures with circular points, slightly offset
offset = 0.1  # Offset to prevent alignment
for i, temp in enumerate(monthly_temps):
    ax.scatter(angles[i] + offset * np.random.randn(), temp, s=100, alpha=0.6, edgecolors='w', label=f'Temp {i // 12 + 2004}' if i % 12 == 0 else "")

# Plot the blue curve for 2015
ax.plot(angles, temp_2015 + [temp_2015[0]], color='blue', linewidth=2, label='2015 Temperatures')  # Close the loop

# Set the labels for each month
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(angles)
ax.set_xticklabels(month_labels)

# Add title and legend
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

# Save the plot to a PNG file
plt.savefig("novice.png")

# Show the plot
plt.show()