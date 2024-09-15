import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Read the data from the CSV file
data = pd.read_csv('data.csv')

# Step 3: Filter and prepare the data
# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the first of each month
first_of_month = data[data['Date'].dt.day == 1]

# Filter data for the specified date range
filtered_data = first_of_month[(first_of_month['Date'] >= '2004-01-01') & (first_of_month['Date'] <= '2015-08-01')]

# Extract the highest temperature for each month
monthly_max_temps = filtered_data.groupby([filtered_data['Date'].dt.year, filtered_data['Date'].dt.month]).agg({'Temperature': 'max'}).reset_index(drop=False)

# Step 4: Plot the data
# Create a polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Define the angles for each month (12 sectors)
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Add a small offset to prevent alignment along a single radial line
offset = 0.1

# Plot the temperature data
for year in monthly_max_temps[0].unique():
    year_data = monthly_max_temps[monthly_max_temps[0] == year]
    temps = year_data['Temperature'].values
    months = year_data[1].values - 1  # Convert to 0-based index
    angles_with_offset = [angles[m] + offset * (i % 2) for i, m in enumerate(months)]
    ax.scatter(angles_with_offset, temps, label=str(year))

# Highlight the data points for the year 2015 with a blue curve
data_2015 = monthly_max_temps[monthly_max_temps[0] == 2015]
temps_2015 = data_2015['Temperature'].values
months_2015 = data_2015[1].values - 1  # Convert to 0-based index
angles_2015 = [angles[m] for m in months_2015]
ax.plot(angles_2015, temps_2015, 'b-', label='2015')

# Add month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(angles)
ax.set_xticklabels(month_labels)

# Add a legend positioned on the right side of the diagram
ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

# Add a title
plt.title('Monthly Highest Temperature in Amherst (2004-2015)')

# Save the plot to a PNG file
plt.savefig('plot.png')

# Show the plot
plt.show()