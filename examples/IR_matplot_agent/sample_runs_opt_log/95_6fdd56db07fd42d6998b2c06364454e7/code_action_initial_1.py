# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the Data
data = pd.read_csv("data.csv")

# Step 3: Data Preparation
# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the years 2004 to 2015
data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Extract the month and year
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Get the highest temperature for the first of each month
monthly_max = data.groupby(['Year', 'Month'])['Temperature'].max().reset_index()

# Step 4: Extract Data for Plotting
# Create a list of months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Prepare data for plotting
temperatures = []
for month in range(1, 13):
    temp = monthly_max[(monthly_max['Month'] == month) & (monthly_max['Year'] >= 2004)]['Temperature']
    temperatures.append(temp.max() if not temp.empty else np.nan)

# Extract 2015 data for the blue curve
temperatures_2015 = monthly_max[monthly_max['Year'] == 2015]['Temperature'].values

# Ensure that we have 12 values for the 2015 data by filling in NaNs for missing months
temperatures_2015_full = [np.nan] * 12
for month in range(1, 13):
    if month in monthly_max[monthly_max['Year'] == 2015]['Month'].values:
        temperatures_2015_full[month - 1] = temperatures_2015[np.where(monthly_max['Month'] == month)[0][0]]

# Step 5: Create the Polar Plot
# Create a polar plot
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)  # 12 sectors for 12 months
radii = temperatures  # Temperatures for each month

# Create the figure and polar axis
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})

# Plot the circular points with slight offset
offset = 0.1  # Offset to prevent alignment
ax.scatter(theta + offset, radii, color='red', s=100, label='Monthly Max Temp')

# Connect the 2015 data with a blue curve
ax.plot(theta, temperatures_2015_full, color='blue', linewidth=2, label='2015 Data')

# Set the labels for each month
ax.set_xticks(theta)
ax.set_xticklabels(months)

# Set the title and legend
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')
ax.legend(loc='upper right')

# Save the plot to a PNG file
plt.savefig("novice.png")

# Show the plot
plt.show()