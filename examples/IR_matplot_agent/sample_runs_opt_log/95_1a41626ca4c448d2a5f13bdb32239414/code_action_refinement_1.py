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
data = data[(data['Year'] >= 2004) & (data['Year'] <= 2015)]

# Extract the first of each month and find the maximum temperature
monthly_data = data[data['Date'].dt.day == 1].groupby(['Year', data['Date'].dt.month])['Temperature'].max().reset_index()

# Step 4: Prepare Data for Polar Plot
# Create a list of angles for each month
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Extract temperatures for the year 2015
temp_2015 = monthly_data[monthly_data['Year'] == 2015]['Temperature'].tolist()

# Prepare the temperatures for all years
monthly_temps = []
for month in range(1, 13):
    monthly_temps.append(monthly_data[monthly_data['Date'] == month]['Temperature'].tolist())

# Flatten the list of temperatures for plotting
monthly_temps = [temp for sublist in monthly_temps for temp in sublist]

# Step 5: Create the Polar Plot
# Create polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot the temperatures with circular points
for i, temp in enumerate(monthly_temps):
    ax.scatter(angles[i % 12] + np.random.uniform(-0.1, 0.1), temp, s=100, alpha=0.6, edgecolors='w', label=f'Temp {i // 12 + 2004}' if i % 12 == 0 else "")

# Ensure temp_2015 has 12 values for plotting
temp_2015_full = [temp_2015[i] if i < len(temp_2015) else np.nan for i in range(12)]

# Plot the blue curve for 2015
ax.plot(angles, temp_2015_full, color='blue', linewidth=2, label='2015 Temperatures')

# Set the labels for each month
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(angles)
ax.set_xticklabels(month_labels)

# Add title and legend
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()