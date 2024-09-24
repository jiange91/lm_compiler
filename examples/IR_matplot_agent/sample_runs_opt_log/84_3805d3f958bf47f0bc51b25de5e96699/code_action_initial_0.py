import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from CSV
data = pd.read_csv('data.csv')

# Extract data for phase boundaries
temp_gas_liquid = data.iloc[:, 0]  # Column 1: Temperature for gas-liquid boundary
pressure_gas_liquid = data.iloc[:, 1]  # Column 2: Pressure for gas-liquid boundary
temp_solid_liquid = data.iloc[:, 2]  # Column 3: Temperature for solid-liquid boundary
pressure_solid_liquid = data.iloc[:, 3]  # Column 4: Pressure for solid-liquid boundary

# Create the figure and axis
plt.figure(figsize=(10, 6))
plt.title('Phase Diagram of Water')
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (Pa)')

# Set the limits for the axes
plt.xlim(-50, 100)  # Temperature range
plt.ylim(1e2, 1e7)  # Pressure range (log scale)

# Plot the gas-liquid boundary
plt.plot(temp_gas_liquid, pressure_gas_liquid, label='Gas-Liquid Boundary', color='blue')

# Plot the solid-liquid boundary
plt.plot(temp_solid_liquid, pressure_solid_liquid, label='Solid-Liquid Boundary', color='green')

# Mark the triple point
plt.plot(0.01, 611.657, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')

# Mark the critical point
plt.plot(374.0, 22.064e6, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Draw vertical lines for freezing and boiling points
plt.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')
plt.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Fill the solid phase
plt.fill_betweenx(np.linspace(0, 611.657, 100), -50, 0, color='lightblue', alpha=0.5, label='Solid Phase')

# Fill the liquid phase
plt.fill_betweenx(np.linspace(611.657, 22.064e6, 100), 0, 100, color='lightgreen', alpha=0.5, label='Liquid Phase')

# Fill the gas phase
plt.fill_betweenx(np.linspace(22.064e6, 1e7, 100), 100, 150, color='lightcoral', alpha=0.5, label='Gas Phase')

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Set logarithmic scale for pressure
plt.yscale('log')

# Add legend
plt.legend()

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()