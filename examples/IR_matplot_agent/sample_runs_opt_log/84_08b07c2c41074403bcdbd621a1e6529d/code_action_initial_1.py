import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from CSV
data = pd.read_csv('data.csv', header=None)

# Extracting data for phase boundaries
temp_gas_liquid = data[0].astype(float)  # Temperature for gas-liquid boundary
pressure_gas_liquid = data[1].astype(float)  # Pressure for gas-liquid boundary
temp_solid_liquid = data[2].astype(float)  # Temperature for solid-liquid boundary
pressure_solid_liquid = data[3].astype(float)  # Pressure for solid-liquid boundary

# Create the figure and axis
plt.figure(figsize=(10, 6))
plt.title('Phase Diagram of Water')
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (Pa)')

# Set the limits for the axes
plt.xlim(-50, 200)  # Adjust as necessary
plt.ylim(1e3, 25e6)  # 0 to 25 MPa in Pascals, starting from 1e3 to avoid log(0)

# Plot the gas-liquid boundary
plt.plot(temp_gas_liquid, pressure_gas_liquid, label='Gas-Liquid Boundary', color='blue')

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
plt.fill_betweenx(np.linspace(0, 611.657, 100), -50, 0, color='lightblue', alpha=0.5, label='Solid Phase')

# Fill the liquid phase
plt.fill_betweenx(np.linspace(611.657, 22.064e6, 100), 0, 100, color='lightgreen', alpha=0.5, label='Liquid Phase')

# Fill the gas phase
plt.fill_betweenx(np.linspace(22.064e6, 25e6, 100), 100, 200, color='lightcoral', alpha=0.5, label='Gas Phase')

# Set y-axis to logarithmic scale
plt.yscale('log')

# Add grid and legend
plt.grid(True)
plt.legend()

# Save the plot to a PNG file
plt.savefig("novice.png")

# Show the plot
plt.show()