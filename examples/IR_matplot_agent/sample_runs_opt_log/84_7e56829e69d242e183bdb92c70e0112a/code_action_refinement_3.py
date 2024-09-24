import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from CSV, ensuring to skip the header
data = pd.read_csv('data.csv')

# Check if the expected columns exist
if data.shape[1] < 5:
    raise ValueError("The data must contain at least 5 columns for the phase boundaries.")

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Pressure (Pa)')
ax.set_title('Phase Diagram of Water')

# Set x and y limits
ax.set_xlim(-50, 200)
ax.set_ylim(100, 1e7)

# Set ticks for temperature and pressure
ax.set_xticks(np.arange(-50, 201, 10))
ax.set_yticks([100, 1000, 10000, 100000, 1000000, 10000000])
ax.set_yticklabels(['100 Pa', '1 kPa', '10 kPa', '100 kPa', '1 MPa', '10 MPa'])

# Assuming columns 1 and 2 are for gas-liquid boundary
ax.plot(data.iloc[:, 1], data.iloc[:, 2], label='Gas-Liquid Boundary', color='blue')
# Assuming columns 3 and 4 are for solid-liquid boundary
ax.plot(data.iloc[:, 3], data.iloc[:, 4], label='Solid-Liquid Boundary', color='green')

# Mark special points
# Triple point
ax.plot(0, 611.657, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')
# Critical point
ax.plot(100, 22.064e6, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Draw freezing and boiling points
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')
ax.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Color the phases
ax.fill_betweenx([0, 611.657], -50, 0, color='lightblue', alpha=0.5, label='Solid Phase')
ax.fill_betweenx([611.657, 22.064e6], 0, 100, color='lightgreen', alpha=0.5, label='Liquid Phase')
ax.fill_betweenx([22.064e6, 1e7], 100, 200, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add a grid and logarithmic scale
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend
ax.legend()

# Save the plot
plt.savefig('novice_final.png', dpi=300)
plt.show()