# Step 1: Import Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Read and Prepare the Data
# Load the data from "data.csv"
data = pd.read_csv("data.csv")

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data to include only the first of each month
first_of_month = data[data['Date'].dt.day == 1]

# Extract the highest temperature for each month from January 1, 2004, to August 1, 2015
filtered_data = first_of_month[(first_of_month['Date'] >= '2004-01-01') & (first_of_month['Date'] <= '2015-08-01')]

# Extract data for the year 2015
data_2015 = filtered_data[filtered_data['Date'].dt.year == 2015]

# Step 3: Plot the Data
# Create a polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Define the months and corresponding angles
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
angles = np.linspace(0, 2 * np.pi, len(months), endpoint=False).tolist()

# Plot temperature data for each month with circular points
for year in filtered_data['Date'].dt.year.unique():
    year_data = filtered_data[filtered_data['Date'].dt.year == year]
    temperatures = year_data['Temperature'].tolist()
    # Ensure the angles list matches the length of temperatures
    offset_angles = [angles[i % len(angles)] + np.random.uniform(-0.05, 0.05) for i in range(len(temperatures))]
    ax.scatter(offset_angles, temperatures, label=str(year), alpha=0.75)

# Connect the data points for the year 2015 with a blue curve
temperatures_2015 = data_2015['Temperature'].tolist()
ax.plot(angles[:len(temperatures_2015)], temperatures_2015, 'b-', label='2015', linewidth=2)

# Add month labels
ax.set_xticks(angles)
ax.set_xticklabels(months)

# Add a legend positioned on the right side of the diagram
ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

# Add a title
plt.title("Monthly Highest Temperature in Amherst (2004-2015)")

# Save the plot to a file
plt.savefig('plot.png')

# Show the plot
plt.show()