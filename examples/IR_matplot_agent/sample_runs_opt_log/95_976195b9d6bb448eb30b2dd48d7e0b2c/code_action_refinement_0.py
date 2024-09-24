import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the data
data = pd.read_csv("data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data[(data['Date'] >= '2004-01-01') & (data['Date'] <= '2015-08-01')]

# Step 2: Extract highest temperatures for the first of each month
monthly_data = data[data['Date'].dt.day == 1]
monthly_data['Month'] = monthly_data['Date'].dt.month
monthly_max = monthly_data.groupby('Month')['Temperature'].max().reset_index()

# Step 3: Prepare data for plotting
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()
temperatures = monthly_max['Temperature'].tolist()
temperatures += temperatures[:1]  # Close the loop
angles += angles[:1]  # Close the loop

# Step 4: Create the polar plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, temperatures, color='gray', alpha=0.25)

# Step 5: Plot 2015 data
data_2015 = data[data['Date'].dt.year == 2015]
monthly_2015 = data_2015[data_2015['Date'].dt.day == 1]
monthly_2015_max = monthly_2015.groupby(monthly_2015['Date'].dt.month)['Temperature'].max().reset_index()
temperatures_2015 = monthly_2015_max['Temperature'].tolist()
temperatures_2015 += temperatures_2015[:1]  # Close the loop

ax.plot(angles, temperatures_2015, color='blue', marker='o', label='Highest Temperature (2015)', markersize=8)

# Step 6: Add labels and title
ax.set_xticks(angles[:-1])
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_title("Monthly Highest Temperature in Amherst (2004-2015)", va='bottom')

# Step 7: Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Step 8: Save the plot
plt.savefig("novice_final.png")
plt.show()