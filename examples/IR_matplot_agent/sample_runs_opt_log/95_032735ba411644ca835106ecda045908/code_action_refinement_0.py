import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("data.csv")

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter for the required date range
data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Extract the month and year
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Get the highest temperature for the first of each month
monthly_max = data.groupby(['Year', 'Month'])['Temperature'].max().reset_index()

# Extract data for 2015
data_2015 = monthly_max[monthly_max['Year'] == 2015]

# Prepare the polar coordinates
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)  # 12 sectors for 12 months
r = data_2015['Temperature'].values

# Offset the points slightly to avoid alignment
offset = np.pi / 12  # Adjust this value for more or less offset
theta += offset

# Create the polar plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

# Plot the data points
ax.plot(theta, r, marker='o', color='blue', label='2015 Temperatures')

# Add circular points
ax.scatter(theta, r, color='blue')

# Set the labels for the months
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
ax.set_xticklabels(month_labels)

# Add a title
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')

# Add a legend
ax.legend(loc='upper right')

# Save the plot
plt.savefig("novice_final.png")
plt.show()