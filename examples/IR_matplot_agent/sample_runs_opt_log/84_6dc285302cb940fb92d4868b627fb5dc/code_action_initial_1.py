import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('data.csv', header=None)

# Ensure the data is numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Convert temperature from Celsius to Kelvin
temp_solid_liquid_gas = data[0] + 273.15  # Column 1
pressure_solid_liquid_gas = data[1]        # Column 2

temp_solid_liquid = data[2] + 273.15       # Column 3
pressure_solid_liquid = data[3]             # Column 4

# Create the figure and axis
plt.figure(figsize=(10, 6))
plt.title('Phase Diagram of Water')
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (Pa)')
plt.yscale('log')  # Set the y-axis to logarithmic scale

# Plot the solid-liquid-gas boundary
plt.plot(temp_solid_liquid_gas, pressure_solid_liquid_gas, label='Solid-Liquid-Gas Boundary', color='blue')

# Plot the solid-liquid boundary
plt.plot(temp_solid_liquid, pressure_solid_liquid, label='Solid-Liquid Boundary', color='green')

# Mark the triple point
plt.plot(273.16, 611.657, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')

# Mark the critical point
plt.plot(647.396, 22.064e6, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Draw vertical lines for freezing and boiling points
plt.axvline(x=273.15, color='red', linestyle='--', label='Freezing Point (0 °C)')
plt.axvline(x=373.15, color='orange', linestyle='--', label='Boiling Point (100 °C)')

# Fill the solid phase
plt.fill_betweenx(y=[1e2, 1e6], x1=0, x2=273.15, color='lightblue', alpha=0.5, label='Solid Phase')

# Fill the liquid phase
plt.fill_betweenx(y=[1e2, 1e6], x1=273.15, x2=373.15, color='lightgreen', alpha=0.5, label='Liquid Phase')

# Fill the gas phase
plt.fill_betweenx(y=[1e2, 1e6], x1=373.15, x2=700, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend
plt.legend()

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()