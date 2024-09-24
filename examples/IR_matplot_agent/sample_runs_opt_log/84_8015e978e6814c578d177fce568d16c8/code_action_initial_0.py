import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
data = pd.read_csv('data.csv')

# Convert temperature to Kelvin
data['Temperature_K'] = data.iloc[:, 0] + 273.15  # Column 1 to Kelvin

# Extract pressure and temperature for phase boundaries
pressure_phase_boundary = data.iloc[:, 1]  # Column 2
temperature_phase_boundary = data['Temperature_K']  # Converted Column 1
solid_liquid_boundary = data.iloc[:, 2] + 273.15  # Column 3 to Kelvin
solid_gas_boundary = data.iloc[:, 3] + 273.15  # Column 4 to Kelvin

# Create the figure and axis
plt.figure(figsize=(10, 6))
plt.title('Phase Diagram of Water')
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (Pa)')
plt.yscale('log')  # Set pressure scale to logarithmic

# Plot the phase boundaries
plt.plot(temperature_phase_boundary, pressure_phase_boundary, label='Phase Boundary', color='blue')
plt.plot(solid_liquid_boundary, pressure_phase_boundary, label='Solid-Liquid Boundary', color='green')
plt.plot(solid_gas_boundary, pressure_phase_boundary, label='Solid-Gas Boundary', color='orange')

# Mark the triple point
plt.plot(273.16, 611.657, 'ro')  # Triple point
plt.text(273.16, 611.657, 'Triple Point', fontsize=10, verticalalignment='bottom')

# Mark the critical point
plt.plot(647.396, 22.064 * 1e6, 'ro')  # Critical point (converted to Pa)
plt.text(647.396, 22.064 * 1e6, 'Critical Point', fontsize=10, verticalalignment='bottom')

# Freezing point at 0°C (273.15 K)
plt.axvline(x=273.15, color='red', linestyle='--', label='Freezing Point')

# Boiling point at 100°C (373.15 K)
plt.axvline(x=373.15, color='red', linestyle='--', label='Boiling Point')

# Fill the regions
plt.fill_betweenx(y=[1e3, 1e6], x1=0, x2=273.15, color='lightblue', alpha=0.5, label='Solid Phase')
plt.fill_betweenx(y=[1e3, 1e6], x1=273.15, x2=373.15, color='lightgreen', alpha=0.5, label='Liquid Phase')
plt.fill_betweenx(y=[1e3, 1e6], x1=373.15, x2=1000, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend
plt.legend()

# Save the plot to a PNG file
plt.savefig('novice.png')

# Show the plot
plt.show()