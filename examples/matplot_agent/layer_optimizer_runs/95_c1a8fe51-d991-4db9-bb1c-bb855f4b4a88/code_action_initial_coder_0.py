import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the data
data = pd.read_csv('data.csv')

# Step 2: Filter data for the first of each month
data['Date'] = pd.to_datetime(data['Date'])
first_of_month = data[data['Date'].dt.is_month_start]

# Step 3: Extract the highest temperature for each month
monthly_max_temp = first_of_month.groupby([first_of_month['Date'].dt.year, first_of_month['Date'].dt.month])['Temperature'].max().reset_index()
monthly_max_temp.columns = ['Year', 'Month', 'Temperature']

# Step 4: Separate data for the year 2015
data_2015 = monthly_max_temp[monthly_max_temp['Year'] == 2015]

# Step 5: Prepare the polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Define the angles for each month (12 sectors)
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Add a full circle to close the plot
angles += angles[:1]

# Prepare the temperature data for plotting
temps = monthly_max_temp.pivot(index='Year', columns='Month', values='Temperature')
temps = temps.reindex(columns=range(1, 13)).fillna(0)  # Ensure all months are present
temps = temps.values.T  # Transpose to get months as rows

# Plot each year's data
for year_data in temps:
    year_data = np.append(year_data, year_data[0])  # Close the loop
    ax.plot(angles, year_data, 'o-', linewidth=1, markersize=4)

# Plot the 2015 data with a blue curve
data_2015_temps = data_2015['Temperature'].tolist()
data_2015_temps += data_2015_temps[:1]  # Close the loop
ax.plot(angles, data_2015_temps, 'o-', linewidth=2, markersize=6, color='blue', label='2015')

# Add month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(angles[:-1])
ax.set_xticklabels(month_labels)

# Add a legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Add a title
plt.title('Monthly Highest Temperature in Amherst (2004-2015)')

# Save the plot to a file
plt.savefig('plot.png')

# Show the plot
plt.show()