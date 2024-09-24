import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("data.csv")

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter for the first of each month and the specified date range
filtered_data = data[(data['Date'].dt.day == 1) & 
                     (data['Date'] >= '2004-01-01') & 
                     (data['Date'] <= '2015-08-01')]

# Group by month and get the max temperature
monthly_max = filtered_data.groupby(filtered_data['Date'].dt.month)['Temperature'].max()

# Extract 2015 data for the blue curve
data_2015 = filtered_data[filtered_data['Date'].dt.year == 2015]
monthly_max_2015 = data_2015.groupby(data_2015['Date'].dt.month)['Temperature'].max()

# Create a polar plot
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
radii = monthly_max.reindex(range(1, 13)).values  # Ensure all 12 months are present
radii_2015 = monthly_max_2015.reindex(range(1, 13)).values  # Ensure all 12 months are present

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

# Plot the data points with circular markers
ax.scatter(theta, radii, color='orange', s=100, label='Monthly Max Temp', zorder=5)

# Offset the points slightly
offset = 0.1
ax.scatter(theta + offset, radii, color='orange', s=100)

# Plot the blue curve for 2015
ax.plot(theta, radii_2015, color='blue', linewidth=2, label='2015 Max Temp', zorder=4)

# Add month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(theta)
ax.set_xticklabels(month_labels)

# Add title and legend
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')
ax.legend(loc='upper right')

# Save the plot
plt.savefig("novice_final.png", bbox_inches='tight')
plt.show()