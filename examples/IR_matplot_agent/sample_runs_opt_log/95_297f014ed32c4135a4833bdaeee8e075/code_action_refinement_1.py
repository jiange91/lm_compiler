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
monthly_data = filtered_data.groupby(filtered_data['Date'].dt.month).agg({'Temperature': 'max'}).reset_index()

# Prepare the data for polar plot
months = np.arange(12)
temperatures = np.zeros(12)  # Initialize an array for temperatures
for index, row in monthly_data.iterrows():
    temperatures[row['Date'] - 1] = row['Temperature']  # Fill in the temperatures

# Create polar plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Plot the temperatures
ax.plot(months, temperatures, 'o-', color='red', label='Highest Temp (2004-2015)')

# Extract temperatures for the year 2015
temperatures_2015 = filtered_data[filtered_data['Date'].dt.year == 2015].groupby(filtered_data['Date'].dt.month)['Temperature'].max().values

# Ensure temperatures_2015 has 12 values for plotting
temp_2015_full = np.zeros(12)
for month in range(1, 13):
    if month in monthly_data['Date'].values:
        temp_2015_full[month - 1] = temperatures_2015[month - 1] if month <= len(temperatures_2015) else np.nan

# Plot the blue curve for 2015
ax.plot(months, temp_2015_full, 'o-', color='blue', label='Temperature Curve (2015)')

# Adjust circular points
offset = 0.1  # Adjust this value as needed
ax.scatter(months + offset, temperatures, color='red')

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