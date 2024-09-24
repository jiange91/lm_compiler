import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('data.csv', header=None)

# Extract data for phase boundaries
temp_solid_liquid_gas = data[0]
pressure_solid_liquid_gas = data[1]
temp_solid_liquid = data[2]
pressure_solid_liquid = data[3]

# Create the plot
plt.figure(figsize=(10, 6))
plt.title('Phase Diagram of Water')
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (Pa)')

# Set the x-axis limits
plt.xlim(-50, 200)  # Adjust as necessary for your data
# Set the y-axis limits
plt.ylim(1e2, 1e7)  # Logarithmic scale for pressure

# Plot the solid-liquid-gas boundary
plt.plot(temp_solid_liquid_gas, pressure_solid_liquid_gas, label='Solid-Liquid-Gas Boundary', color='blue')

# Plot the solid-liquid boundary
plt.plot(temp_solid_liquid, pressure_solid_liquid, label='Solid-Liquid Boundary', color='green')

# Mark the triple point
plt.plot(0.01, 611.657, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')

# Mark the critical point
plt.plot(374.0, 22.064e6, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Freezing point at 0°C
plt.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')

# Boiling point at 100°C
plt.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Fill the solid phase
plt.fill_betweenx(y=[1e2, 611.657], x1=-50, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')

# Fill the liquid phase
plt.fill_betweenx(y=[611.657, 22.064e6], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')

# Fill the gas phase
plt.fill_betweenx(y=[22.064e6, 1e7], x1=100, x2=200, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Set y-axis to logarithmic scale
plt.yscale('log')

# Add legend
plt.legend()

# Save the plot to a PNG file
plt.savefig("novice.png")

# Show the plot
plt.show()