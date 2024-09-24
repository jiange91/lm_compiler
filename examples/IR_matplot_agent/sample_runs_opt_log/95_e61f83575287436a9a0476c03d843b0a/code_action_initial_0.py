# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the Data
data = pd.read_csv('data.csv')

# Step 3: Data Preparation
# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the specified date range
mask = (data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')
filtered_data = data.loc[mask]

# Extract month and year
filtered_data['Month'] = filtered_data['Date'].dt.month
filtered_data['Year'] = filtered_data['Date'].dt.year

# Get the highest temperature for the first of each month
monthly_max = filtered_data.groupby(['Year', 'Month'])['Temperature'].max().reset_index()

# Separate data for 2015
data_2015 = monthly_max[monthly_max['Year'] == 2015]

# Prepare polar coordinates
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()
angles += angles[:1]  # Close the circle
monthly_max_temp = monthly_max['Temperature'].tolist()
monthly_max_temp += monthly_max_temp[:1]  # Close the circle for plotting

# Create the polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
ax.plot(angles, monthly_max_temp, color='orange', marker='o', label='Monthly Max Temp')

# Prepare the 2015 data for plotting
temp_2015 = [data_2015[data_2015['Month'] == month]['Temperature'].values[0] if month in data_2015['Month'].values else 0 for month in range(1, 13)]
temp_2015 += [temp_2015[0]]  # Close the circle for 2015 data
ax.plot(angles, temp_2015, color='blue', marker='o', label='2015 Temp')

# Set the labels for each month
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(angles[:-1])
ax.set_xticklabels(month_labels)

# Add a title and legend
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Save the plot to a file
plt.savefig('novice.png')

# Display the plot
plt.show()