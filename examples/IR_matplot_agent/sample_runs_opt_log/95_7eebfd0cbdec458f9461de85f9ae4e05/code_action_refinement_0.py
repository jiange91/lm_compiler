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

# Prepare the data for polar plot
months = np.arange(12)
temperatures = monthly_data['Temperature'].values

# Create polar plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Plot the temperatures with circular points
ax.plot(months, temperatures, color='blue', marker='o', label='Temperature Trend (2015)')
ax.scatter(months, temperatures, color='red', label='Monthly Max Temp (2015)', zorder=5)

# Offset points slightly
for i in range(len(months)):
    ax.text(months[i], temperatures[i] + 1, str(temperatures[i]), horizontalalignment='center')

# Set the labels for the months
ax.set_xticks(months)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Add title and legend
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')
ax.legend(loc='upper right')

# Save the plot
plt.savefig("novice_final.png")
plt.show()