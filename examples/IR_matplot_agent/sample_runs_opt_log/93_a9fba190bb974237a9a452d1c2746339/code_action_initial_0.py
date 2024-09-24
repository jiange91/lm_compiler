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
ax.plot(bond_1_2, bond_1_14, zs=0, zdir='z', color='blue', alpha=0.3, linewidth=2, label='XY Projection')
ax.plot(bond_1_2, zs=0, zdir='y', y=bond_1_14, color='red', alpha=0.3, linewidth=2, label='XZ Projection')
ax.plot(zs=0, y=bond_1_14, z=bond_1_2, zdir='x', color='green', alpha=0.3, linewidth=2, label='YZ Projection')

# Label the axes
ax.set_xlabel('Bond 1-2')
ax.set_ylabel('Bond 1-14')
ax.set_zlabel('Total Energy / Eh')

# Format the z-axis labels for clarity
ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

# Add a color bar
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
cbar.set_label('Time (fs)')

# Add a legend for the projection lines
ax.legend()

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()