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

# Filter data for the year 2015
data_2015 = filtered_data[filtered_data['Year'] == 2015]

# Group by month and get the highest temperature for each month in 2015
monthly_max_temps_2015 = data_2015.groupby('Month')['Temperature'].max().reset_index()

# Step 4: Prepare Data for Polar Plot
# Create angles for each month (0 to 2*pi)
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Get the temperature values
temperatures = monthly_max_temps_2015['Temperature'].tolist()

# To create a closed loop for the polar plot, append the first temperature to the end
temperatures += temperatures[:1]
angles += angles[:1]

# Step 5: Create the Polar Plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot the temperatures
ax.plot(angles, temperatures, color='blue', linewidth=2, label='Temperature 2015')

# Add circular points for each temperature
ax.scatter(angles[:-1], temperatures[:-1], color='red', s=100, zorder=5)

# Set the title
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')

# Set the labels for each month
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(angles[:-1])
ax.set_xticklabels(month_labels)

# Add a legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Save the plot to a PNG file
plt.savefig("plot_final.png")

# Show the plot
plt.show()