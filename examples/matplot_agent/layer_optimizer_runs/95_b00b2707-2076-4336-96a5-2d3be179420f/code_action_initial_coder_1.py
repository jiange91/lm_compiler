# Step 1: Import Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the Data
data = pd.read_csv('data.csv')

# Step 3: Filter and Prepare the Data
# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the first of each month
first_of_month = data[data['Date'].dt.is_month_start]

# Filter data between January 1, 2004, and August 1, 2015
filtered_data = first_of_month[(first_of_month['Date'] >= '2004-01-01') & (first_of_month['Date'] <= '2015-08-01')]

# Extract year, month, and temperature
filtered_data['Month'] = filtered_data['Date'].dt.month
filtered_data['Year'] = filtered_data['Date'].dt.year

# Step 4: Create the Polar Plot
# Set up the polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Define the angles for each month (12 sectors)
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Add a full circle to close the plot
angles += angles[:1]

# Define month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot data points
for year in filtered_data['Year'].unique():
    year_data = filtered_data[filtered_data['Year'] == year]
    
    # Initialize temperatures with NaNs for all 12 months
    temperatures = [np.nan] * 12
    
    # Fill in the temperatures for the available months
    for month in year_data['Month'].unique():
        temperatures[month - 1] = year_data[year_data['Month'] == month]['Temperature'].values[0]
    
    # Close the loop
    temperatures += temperatures[:1]
    
    # Offset points slightly to prevent alignment along a single radial line
    offset_angles = [angle + np.random.uniform(-0.05, 0.05) for angle in angles]

    if year == 2015:
        ax.plot(offset_angles, temperatures, 'o-', label=str(year), color='blue')
    else:
        ax.plot(offset_angles, temperatures, 'o', label=str(year))

# Add month labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(month_labels)

# Add legend
ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

# Add title
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')

# Save plot to a file
plot_file_name = 'plot.png'
plt.savefig(plot_file_name)

# Show plot
plt.show()