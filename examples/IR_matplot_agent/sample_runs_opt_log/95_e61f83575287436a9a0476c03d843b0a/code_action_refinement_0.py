import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv("data.csv")

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter for the first of each month and the specified date range
filtered_data = data[(data['Date'].dt.day == 1) & 
                     (data['Date'] >= '2004-01-01') & 
                     (data['Date'] <= '2015-08-01')]

# Extract the highest temperatures for each month
monthly_data = filtered_data.groupby(filtered_data['Date'].dt.month).max()

# Separate the data for the year 2015
temperatures_2015 = monthly_data[monthly_data['Date'].dt.year == 2015]['Temperature']

# Prepare the data for polar plot
months = np.arange(12)
temperatures = monthly_data['Temperature'].values

# Create polar plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Plot the temperatures
ax.plot(months, temperatures, marker='o', label='Monthly Max Temp', color='orange')

# Offset the points slightly
offset = 0.1
ax.scatter(months + offset, temperatures_2015, color='blue', s=100, zorder=5)

# Set month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(months)
ax.set_xticklabels(month_labels)

# Add title and legend
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')
ax.legend(loc='upper right')

# Save the plot
plt.savefig("novice_final.png")
plt.show()