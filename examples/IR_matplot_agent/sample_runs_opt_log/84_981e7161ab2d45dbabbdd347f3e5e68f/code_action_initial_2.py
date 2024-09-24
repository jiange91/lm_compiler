import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from CSV
data = pd.read_csv('data.csv', header=None)

# Extracting data for phase boundaries
temp_phase_boundary = data.iloc[:, 0]  # Temperature for solid-liquid-gas boundary
pressure_phase_boundary = data.iloc[:, 1]  # Pressure for solid-liquid-gas boundary

temp_solid_liquid_boundary = data.iloc[:, 2]  # Temperature for solid-liquid boundary
pressure_solid_liquid_boundary = data.iloc[:, 3]  # Pressure for solid-liquid boundary

# Create the figure and axis
plt.figure(figsize=(10, 6))
plt.title('Phase Diagram of Water')
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (Pa)')

# Set the limits for the axes
plt.xlim(-50, 100)  # Temperature range in Celsius
plt.ylim(1e2, 1e7)  # Pressure range in Pascals

# Plot the phase boundaries
plt.plot(temp_phase_boundary, pressure_phase_boundary, label='Solid-Liquid-Gas Boundary', color='blue')
plt.plot(temp_solid_liquid_boundary, pressure_solid_liquid_boundary, label='Solid-Liquid Boundary', color='green')

# Mark the triple point
plt.plot(0.01, 611.657, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')

# Mark the critical point
plt.plot(374.0, 22.064e6, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Draw vertical lines for freezing and boiling points
plt.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')
plt.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Fill the regions for solid, liquid, and gas
plt.fill_betweenx(y=[1e2, 611.657], x1=-50, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')
plt.fill_betweenx(y=[611.657, 22.064e6], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')
plt.fill_betweenx(y=[22.064e6, 1e7], x1=100, x2=150, color='lightyellow', alpha=0.5, label='Gas Phase')

# Set logarithmic scale for pressure
plt.yscale('log')

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend
plt.legend()

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()