import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Prepare the data
data['Date'] = pd.to_datetime(data['Date'])
first_of_month = data[data['Date'].dt.is_month_start]
filtered_data = first_of_month[(first_of_month['Date'] >= '2004-01-01') & (first_of_month['Date'] <= '2015-08-01')]
filtered_data['Month'] = filtered_data['Date'].dt.month
filtered_data['Year'] = filtered_data['Date'].dt.year

# Set up the polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()
angles += angles[:1]
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot data points
for year in filtered_data['Year'].unique():
    year_data = filtered_data[filtered_data['Year'] == year]
    temperatures = [np.nan] * 12

    for month in year_data['Month'].unique():
        temperatures[month - 1] = year_data[year_data['Month'] == month]['Temperature'].values[0]
    
    temperatures += temperatures[:1]
    offset_angles = [angle + np.random.uniform(-0.05, 0.05) for angle in angles]

    if year == 2015:
        ax.plot(offset_angles, temperatures, 'o-', label=str(year), color='blue')
    else:
        ax.plot(offset_angles, temperatures, 'o', label=str(year))

# Add month labels and legend
ax.set_xticks(angles[:-1])
ax.set_xticklabels(month_labels)
ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

# Add title
ax.set_title('Monthly Highest Temperature in Amherst (2004-2015)', va='bottom')

# Save and show the plot
plot_file_name = 'plot_final.png'
plt.savefig(plot_file_name)
plt.show()