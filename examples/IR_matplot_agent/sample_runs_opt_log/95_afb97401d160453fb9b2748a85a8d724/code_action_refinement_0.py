import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")

# Filter data for the first of each month and the specified date range
data['Date'] = pd.to_datetime(data['Date'])
data = data[(data['Date'].dt.day == 1) & (data['Date'].dt.year >= 2004) & (data['Date'].dt.year <= 2015)]

# Extract month and year
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Get highest temperatures for each month
monthly_data = data.loc[data.groupby(['Year', 'Month'])['Temperature'].idxmax()]

# Prepare polar plot
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
r = [monthly_data[monthly_data['Month'] == i]['Temperature'].max() for i in range(1, 13)]

# Offset points slightly
offset = np.random.uniform(-0.5, 0.5, size=len(r))
r = np.array(r) + offset

# Create polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Plot data points
for year in range(2004, 2016):
    year_data = monthly_data[monthly_data['Year'] == year]
    ax.scatter(np.radians(year_data['Month'] * 30), year_data['Temperature'], label=f'Temp {year}', s=100)

# Highlight 2015 data with a blue curve
data_2015 = monthly_data[monthly_data['Year'] == 2015]
ax.plot(np.radians(data_2015['Month'] * 30), data_2015['Temperature'], color='blue', label='2015 Temperatures', linewidth=2)

# Set month labels
ax.set_xticks(np.radians(np.arange(0, 360, 30)))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Add title and legend
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')
ax.legend(loc='upper right')

# Save the plot
plt.savefig("novice_final.png")
plt.show()