# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the Data
data = pd.read_csv("data.csv")

# Step 3: Data Preparation
# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the required date range
filtered_data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Extract month and year
filtered_data['Month'] = filtered_data['Date'].dt.month
filtered_data['Year'] = filtered_data['Date'].dt.year

# Group by month and get the highest temperature for each month
monthly_max_temps = filtered_data.groupby('Month')['Temperature'].max().reset_index()

# Step 4: Prepare Data for Polar Plot
# Prepare angles for each month (0 to 2*pi)
angles = np.linspace(0, 2 * np.pi, len(monthly_max_temps), endpoint=False).tolist()

# Append the first value to the end to close the circle
temperatures = monthly_max_temps['Temperature'].tolist()
temperatures += temperatures[:1]
angles += angles[:1]

# Step 5: Create the Polar Plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot the data points
ax.plot(angles, temperatures, color='blue', linewidth=2, label='2004-2015 Data')

# Add circular points for the temperature data
ax.scatter(angles[:-1], temperatures[:-1], color='red', s=100, zorder=5)

# Set the title and labels
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')
ax.set_xticks(angles[:-1])  # Set the ticks for each month
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Add a legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Save the plot to a file
plt.savefig('plot.png')

# Show the plot
plt.show()