import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Import the data
data = pd.read_csv('data.csv')

# Step 2: Extract the necessary columns
temp1 = data.iloc[:, 0]  # Temperature for solid-liquid-gas boundary
pressure1 = data.iloc[:, 1]  # Pressure for solid-liquid-gas boundary
temp2 = data.iloc[:, 2]  # Temperature for solid-liquid boundary
pressure2 = data.iloc[:, 3]  # Pressure for solid-liquid boundary

# Step 3: Set up the plot
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Pressure (Pa)')
ax.set_title('Phase Diagram of Water')

# Step 4: Plot the phase boundaries
ax.plot(temp1, pressure1, label='Solid-Liquid-Gas Boundary', color='blue')
ax.plot(temp2, pressure2, label='Solid-Liquid Boundary', color='green')

# Step 5: Mark special points
triple_point = (273.16, 611.657)
critical_point = (647.396, 22.064e6)
ax.plot(*triple_point, 'go', label='Triple Point (273.16 K, 611.657 Pa)')
ax.plot(*critical_point, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Step 6: Draw vertical lines for freezing and boiling points
ax.axvline(x=273.15, color='red', linestyle='--', label='Freezing Point (0°C)')
ax.axvline(x=373.15, color='red', linestyle='--', label='Boiling Point (100°C)')

# Step 7: Label and color the regions
# Note: This is a simplified approach to coloring regions. For more accurate coloring, you may need to use polygons.
ax.fill_between(temp1, pressure1, pressure2, where=(pressure1 > pressure2), color='blue', alpha=0.3, label='Solid')
ax.fill_between(temp1, pressure1, pressure2, where=(pressure1 < pressure2), color='green', alpha=0.3, label='Liquid')
ax.fill_between(temp1, pressure1, 1e9, where=(pressure1 < 1e9), color='yellow', alpha=0.3, label='Gas')

# Step 8: Add grid and finalize the plot
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()

# Save the plot to a file
plt.savefig('plot.png')

# Show the plot
plt.show()