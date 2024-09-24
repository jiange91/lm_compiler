import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("data.csv")

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter for the first of each month and the specified date range
data = data[(data['Date'].dt.day == 1) & 
            (data['Date'] >= '2004-01-01') & 
            (data['Date'] <= '2015-08-01')]

# Extract monthly data
monthly_data = data.groupby(data['Date'].dt.to_period('M')).max().reset_index()

# Prepare polar coordinates
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)

# Prepare the polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot each year's data
for year in range(2004, 2016):
    year_data = monthly_data[monthly_data['Date'].dt.year == year]
    offset = np.random.uniform(-0.1, 0.1, size=len(year_data))
    ax.scatter(theta + offset, year_data['Temperature'], label=f'Temp {year}', alpha=0.6)

    # Highlight 2015 with a blue curve
    if year == 2015:
        ax.plot(theta, year_data['Temperature'], color='blue', linewidth=2, label='2015 Temperatures')

# Customize the plot
ax.set_xticks(theta)
ax.set_xticklabels(months)
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Add grid lines
ax.yaxis.grid(True)

# Save the plot
plt.savefig("novice_final.png", bbox_inches='tight')
plt.show()