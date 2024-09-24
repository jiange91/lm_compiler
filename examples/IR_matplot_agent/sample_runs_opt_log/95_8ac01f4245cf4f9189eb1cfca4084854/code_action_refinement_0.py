import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv("data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Step 2: Extract highest temperatures for the first of each month
monthly_data = data[data['Date'].dt.day == 1]
monthly_data['Month'] = monthly_data['Date'].dt.month
monthly_max = monthly_data.groupby('Month')['Temperature'].max().reset_index()

# Step 3: Prepare the polar plot
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
r = monthly_max['Temperature'].values

# Offset points slightly
offset = np.random.uniform(-1, 1, size=r.shape) * 0.5
r += offset

# Step 4: Create the polar plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
ax.plot(theta, r, marker='o', linestyle='-', color='orange', label='Monthly Avg Temp')

# Highlight 2015 data
data_2015 = data[data['Date'].dt.year == 2015]
monthly_2015 = data_2015[data_2015['Date'].dt.day == 1]
monthly_2015_max = monthly_2015.groupby(monthly_2015['Date'].dt.month)['Temperature'].max().values
ax.plot(theta, monthly_2015_max, marker='o', linestyle='-', color='blue', label='2015 Temp')

# Step 5: Customize the plot
ax.set_xticks(theta)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')
ax.legend(loc='upper right')

# Step 6: Save the plot
plt.savefig("novice_final.png")
plt.show()