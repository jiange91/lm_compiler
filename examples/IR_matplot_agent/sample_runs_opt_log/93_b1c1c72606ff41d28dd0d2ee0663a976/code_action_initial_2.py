import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Load the data from the CSV file
data = pd.read_csv('data.csv')

# Extract the relevant columns
t = data['t']  # Time
bond_1_2 = data['bond 1-2']  # Bond length 1-2
bond_1_14 = data['bond 1-14']  # Bond length 1-14
tot_energy = data['tot energy / Eh']  # Total energy in Hartrees

# Create a figure for the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Normalize the time for color mapping
norm = plt.Normalize(t.min(), t.max())
colors = cm.jet(norm(t))  # Use the jet color map

# Plot the points with larger, semi-transparent markers
scatter = ax.scatter(bond_1_2, bond_1_14, tot_energy, c=colors, s=100, alpha=0.6)

# Connect the points with lines to show the trajectory
ax.plot(bond_1_2, bond_1_14, tot_energy, color='gray', alpha=0.5)

# Projection lines
ax.plot(bond_1_2, bond_1_14, zs=0, zdir='z', color='blue', alpha=0.3)  # XY projection
ax.plot(bond_1_2, zs=tot_energy, zdir='y', color='red', alpha=0.3)  # XZ projection
ax.plot(x=bond_1_2, y=bond_1_14, zs=tot_energy, zdir='x', color='green', alpha=0.3)  # YZ projection

# Set axis labels
ax.set_xlabel('Bond 1-2')
ax.set_ylabel('Bond 1-14')
ax.set_zlabel('Total Energy / Eh')

# Format the z-axis labels for clarity
ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

# Add a color bar for the time progression
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
cbar.set_label('Time (fs)')

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()