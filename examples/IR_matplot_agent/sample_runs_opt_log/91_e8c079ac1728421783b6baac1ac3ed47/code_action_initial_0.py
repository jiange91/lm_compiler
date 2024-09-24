import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from CSV
data = pd.read_csv('data.csv')

# Extract time and frequency data
time = data.iloc[:, 0]  # First column as time
frequencies = data.columns[1:]  # Subsequent columns as frequencies

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define colors and markers for each frequency
colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))  # Use a colormap for distinct colors
y_offset = 0  # Initial y-offset for spacing

for i, freq in enumerate(frequencies):
    amplitude = data[freq]  # Get the amplitude data for the current frequency
    ax.plot(time, np.full_like(time, y_offset), amplitude, color=colors[i], marker='o', label=f'{freq}')
    ax.fill_between(time, 0, amplitude, alpha=0.1, color=colors[i])  # Fill below the line
    y_offset += 1  # Increment y-offset for the next frequency

# Set camera view
ax.view_init(elev=15, azim=-69)  # Set elevation and azimuth

# Set axis labels and limits
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Frequency (Hz)')
ax.set_zlabel('Amplitude (a.u.)')
ax.set_xlim([time.min(), time.max()])  # X-axis limits based on time
ax.set_ylim([-0.5, len(frequencies) - 0.5])  # Y-axis limits for frequencies
ax.set_zlim([data.iloc[:, 1:].min().min(), data.iloc[:, 1:].max().max()])  # Z-axis limits based on amplitude

# Remove y-axis ticks
ax.set_yticks([])  # Remove y-axis ticks

plt.legend()  # Optional: Add a legend to identify frequencies
plt.savefig('novice.png')  # Save the plot to a file
plt.show()  # Display the plot