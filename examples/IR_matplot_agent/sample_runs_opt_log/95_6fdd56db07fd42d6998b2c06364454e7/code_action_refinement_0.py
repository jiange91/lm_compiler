import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")
data['Date'] = pd.to_datetime(data['Date'])
filtered_data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Extract monthly data
monthly_data = filtered_data.groupby(filtered_data['Date'].dt.month).max()

# Prepare polar plot
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
temperatures = monthly_data['Temperature'].values

# Offset data points
offset = 0.1  # Adjust this value for more or less offset
theta_offset = theta + np.random.uniform(-offset, offset, size=theta.shape)

# Plot the data
plt.figure(figsize=(10, 8))
plt.subplot(projection='polar')
plt.plot(theta_offset, temperatures, 'o', color='red', label='Monthly Max Temp')
plt.plot(theta, temperatures, color='blue', label='2015 Data')

# Add month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(theta, month_labels)

# Add legend
plt.legend(loc='upper right')

# Add title
plt.title("Monthly Highest Temperature in Amherst (2004-2015)")

# Save the plot
plt.savefig("novice_final.png")
plt.show()