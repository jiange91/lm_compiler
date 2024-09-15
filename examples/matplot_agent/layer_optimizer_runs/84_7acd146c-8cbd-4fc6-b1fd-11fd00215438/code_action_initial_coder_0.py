import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from 'data.csv'
data = pd.read_csv('data.csv')

# Display the first few rows of the data to understand its structure
print(data.head())

# Convert temperature from Celsius to Kelvin
data['Temperature_K'] = data.iloc[:, 0] + 273.15  # Column 1
data['Pressure_Pa'] = data.iloc[:, 1]  # Column 2
data['Solid_Liquid_Temperature_K'] = data.iloc[:, 2] + 273.15  # Column 3
data['Solid_Liquid_Pressure_Pa'] = data.iloc[:, 3]  # Column 4

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set axis labels
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Pressure (Pa)')

# Set logarithmic scale for pressure
ax.set_yscale('log')

# Set limits for the axes
ax.set_xlim(250, 700)  # Temperature range
ax.set_ylim(1e2, 1e7)  # Pressure range

# Plot the solid-liquid-gas boundary
ax.plot(data['Temperature_K'], data['Pressure_Pa'], label='Solid-Liquid-Gas Boundary', color='blue')

# Plot the solid-liquid boundary
ax.plot(data['Solid_Liquid_Temperature_K'], data['Solid_Liquid_Pressure_Pa'], label='Solid-Liquid Boundary', color='green')

# Mark the triple point
triple_point = (273.16, 611.657)  # (Temperature in K, Pressure in Pa)
ax.plot(*triple_point, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')

# Mark the critical point
critical_point = (647.396, 22.064e6)  # (Temperature in K, Pressure in Pa)
ax.plot(*critical_point, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Freezing point at 0°C (273.15 K)
ax.axvline(x=273.15, color='red', linestyle='--', label='Freezing Point (0°C)')

# Boiling point at 100°C (373.15 K)
ax.axvline(x=373.15, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Fill the regions
ax.fill_betweenx([1e2, 611.657], 250, 273.15, color='lightblue', alpha=0.5, label='Solid Phase')
ax.fill_betweenx([611.657, 22.064e6], 273.15, 373.15, color='lightgreen', alpha=0.5, label='Liquid Phase')
ax.fill_betweenx([22.064e6, 1e7], 373.15, 700, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend
ax.legend()

# Show the plot
plt.title('Phase Diagram of Water')
plt.savefig('plot.png')
plt.show()