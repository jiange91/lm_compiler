import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
data = pd.read_csv('data.csv')

# Extracting the necessary columns
temp_gas_liquid = data.iloc[:, 0]  # Column 1 for gas-liquid boundary
pressure_gas_liquid = data.iloc[:, 1]  # Column 2 for gas-liquid boundary
temp_solid_liquid = data.iloc[:, 2]  # Column 3 for solid-liquid boundary
pressure_solid_liquid = data.iloc[:, 3]  # Column 4 for solid-liquid boundary

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Set labels for the axes
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (Pa)')

# Set the title
plt.title('Phase Diagram of Water')

# Set the y-axis to logarithmic scale
plt.yscale('log')

# Plot the gas-liquid boundary
plt.plot(temp_gas_liquid, pressure_gas_liquid, label='Gas-Liquid Boundary', color='blue')

# Plot the solid-liquid boundary
plt.plot(temp_solid_liquid, pressure_solid_liquid, label='Solid-Liquid Boundary', color='green')

# Mark the triple point
plt.plot(0.01, 611.657, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')

# Mark the critical point
plt.plot(374.0, 22064000, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Draw vertical lines for freezing and boiling points
plt.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0 °C)')
plt.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100 °C)')

# Fill the regions
plt.fill_betweenx([1e2, 1e7], -100, 0, color='lightblue', alpha=0.5, label='Solid Phase')
plt.fill_betweenx([1e2, 1e7], 0, 100, color='lightgreen', alpha=0.5, label='Liquid Phase')
plt.fill_betweenx([1e2, 1e7], 100, 400, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend
plt.legend()

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()