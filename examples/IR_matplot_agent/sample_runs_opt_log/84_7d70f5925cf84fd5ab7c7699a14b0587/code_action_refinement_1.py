import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Check the columns in the data
print(data.columns)

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Pressure (Pa)')
ax.set_title('Phase Diagram of Water')

# Convert Pressure to Logarithmic Scale
ax.set_yscale('log')

# Plot the phase boundaries
ax.plot(data.iloc[:, 0], data.iloc[:, 1], label='Solid-Liquid-Gas Boundary', color='blue')  # Columns 1 and 2
ax.plot(data.iloc[:, 2], data.iloc[:, 3], label='Solid-Liquid Boundary', color='green')  # Columns 3 and 4

# Mark the special points
ax.plot(0, 611.657, 'ro', label='Triple Point', markersize=8)  # Triple Point
ax.plot(374.0, 22064e3, 'ro', label='Critical Point', markersize=8)  # Critical Point

# Draw freezing and boiling points
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')
ax.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Color the phases
ax.fill_betweenx(y=[0, 611.657], x1=-50, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')
ax.fill_betweenx(y=[611.657, 22064e3], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')
ax.fill_betweenx(y=[22064e3, 1e7], x1=100, x2=200, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add a grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend
ax.legend()

# Save the plot
plt.savefig('novice_final.png', dpi=300)
plt.show()