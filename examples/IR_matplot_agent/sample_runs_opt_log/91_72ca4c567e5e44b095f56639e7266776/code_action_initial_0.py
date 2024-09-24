# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 2: Load the Data
data = pd.read_csv('data.csv')  # Load the CSV file

# Step 3: Extract Time and Frequency Data
time = data.iloc[:, 0]  # First column as time
frequencies = data.columns[1:]  # Subsequent columns as frequency data

# Step 4: Create Z-Coordinates for Each Frequency Line
z_values = np.arange(len(frequencies))  # Create an array of z-values for each frequency

# Step 5: Create the 3D Waterfall Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the viewing angle
ax.view_init(elev=15, azim=-69)

# Plot each frequency line
for i, freq in enumerate(frequencies):
    ax.plot(time, np.full_like(time, z_values[i]), data[freq], label=f'Frequency {freq}', marker='o')
    ax.fill_between(time, 0, data[freq], color='lightgrey', alpha=0.5)

# Step 6: Set the Axes Labels
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Frequency (Hz)')
ax.set_zlabel('Amplitude (a.u.)')

# Step 7: Adjust the Y-Axis Limits to Elongate It
ax.set_ylim([-0.5, len(frequencies) - 0.5])  # Adjust y-axis limits

# Step 8: Remove Y-Axis Ticks
ax.set_yticks([])

# Step 9: Save the Plot to a PNG File
plt.savefig('novice.png')

# Step 10: Show the Plot
plt.show()