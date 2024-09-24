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

# Extract Month and Year
filtered_data['Month'] = filtered_data['Date'].dt.month
filtered_data['Year'] = filtered_data['Date'].dt.year

# Prepare Data for Plotting
monthly_data = filtered_data.groupby(['Month', 'Year'])['Temperature'].max().unstack()

# Set up the polar plot
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
theta += np.pi / 12  # Offset for better alignment

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})

# Plot the Data
for year in monthly_data.columns:
    ax.scatter(theta, monthly_data[year], label=year, alpha=0.6)

# Highlight 2015 with a blue curve
ax.plot(theta, monthly_data[2015], color='blue', marker='o', label='2015', linewidth=2)

# Set month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(theta)
ax.set_xticklabels(month_labels)

# Title and legend
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Save the Plot
plt.savefig("novice_final.png", bbox_inches='tight')
plt.show()