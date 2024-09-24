import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('data.csv')
temperature = data['Temperature']  # Assuming column names
pressure = data['Pressure']         # Assuming column names

fig, ax = plt.subplots()
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Pressure (Pa)')

# Plot Phase Boundaries
ax.plot(temperature, pressure, label='Solid-Gas Boundary', color='blue')
ax.plot(temperature, pressure, label='Solid-Liquid Boundary', color='green')

# Mark Special Points
ax.plot(273.16 - 273.15, 611.657, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')
ax.plot(647.396 - 273.15, 22.064 * 1e6, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Draw Freezing and Boiling Points
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')
ax.axvline(x=100, color='red', linestyle='--', label='Boiling Point (100°C)')

# Color the Phases
ax.fill_betweenx(pressure, -10, 0, color='lightblue', alpha=0.5, label='Solid Phase')
ax.fill_betweenx(pressure, 0, 100, color='lightgreen', alpha=0.5, label='Liquid Phase')
ax.fill_betweenx(pressure, 100, 600, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add a Grid and Logarithmic Scale
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add Legend and Title
ax.set_title('Phase Diagram of Water')
ax.legend()

# Save the Plot
plt.savefig('novice_final.png', dpi=300)
plt.show()