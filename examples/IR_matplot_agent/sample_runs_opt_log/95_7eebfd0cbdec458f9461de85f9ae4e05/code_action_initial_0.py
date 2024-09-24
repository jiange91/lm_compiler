# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the Data
data = pd.read_csv('data.csv')

# Step 3: Data Preparation
# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the years 2004 to 2015
data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Extract month and year
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Get the highest temperature for the first of each month
monthly_max = data.groupby(['Year', 'Month'])['Temperature'].max().reset_index()

# Step 4: Extract Data for Plotting
# Create a pivot table to get temperatures for each month
pivot_data = monthly_max.pivot(index='Year', columns='Month', values='Temperature')

# Extract the temperatures for the year 2015
temperatures_2015 = pivot_data.loc[2015]

# Prepare data for plotting
months = np.arange(1, 13)  # Months from 1 to 12
temperatures = temperatures_2015.values  # Temperatures for 2015

# Step 5: Create the Polar Plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})

# Set the angles for each month
angles = np.linspace(0, 2 * np.pi, len(months), endpoint=False).tolist()

# Add the temperatures to the plot
# To avoid alignment, we will slightly offset the temperatures
offset = 0.1  # Offset for circular points
temperatures_offset = temperatures + np.random.uniform(-offset, offset, size=temperatures.shape)

# Plot the circular points
ax.scatter(angles, temperatures_offset, color='red', s=100, label='Monthly Max Temp (2015)', zorder=5)

# Connect the points with a blue curve
ax.plot(angles, temperatures, color='blue', linewidth=2, label='Temperature Trend (2015)', zorder=3)

# Set the labels for each month
ax.set_xticks(angles)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Add a title and legend
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')
ax.legend(loc='upper right')

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()