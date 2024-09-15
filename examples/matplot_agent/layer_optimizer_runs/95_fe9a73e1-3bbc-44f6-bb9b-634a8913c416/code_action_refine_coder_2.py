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

# Filter data for the year 2015
data_2015 = filtered_data[filtered_data['Year'] == 2015]
monthly_max_temps_2015 = data_2015.groupby('Month')['Temperature'].max().reset_index()

# Step 4: Prepare Data for Polar Plot
# Prepare angles for each month (0 to 2*pi)
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Append the first value to the end to close the circle
temperatures = monthly_max_temps['Temperature'].tolist()
temperatures += temperatures[:1]
angles += angles[:1]

# Prepare data for 2015
temperatures_2015 = [np.nan] * 12  # Initialize with NaN for all months
for index, row in monthly_max_temps_2015.iterrows():
    temperatures_2015[row['Month'] - 1] = row['Temperature']
temperatures_2015 += temperatures_2015[:1]

# Apply a slight random offset to the angles
offset_angles = [angle + np.random.uniform(-0.1, 0.1) for angle in angles[:-1]]

# Step 5: Create the Polar Plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot the data points
ax.plot(angles, temperatures, color='blue', linewidth=2, label='2004-2015 Data')

# Plot the 2015 data points
ax.plot(angles, temperatures_2015, color='green', linewidth=2, linestyle='--', label='2015 Data')

# Add circular points for the temperature data with offset
ax.scatter(offset_angles, temperatures[:-1], color='red', s=100, zorder=5)

# Set the title and labels
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')
ax.set_xticks(angles[:-1])  # Set the ticks for each month
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Add a legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Save the plot to a file
plt.savefig('plot_final.png')

# Show the plot
plt.show()