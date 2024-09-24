import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Pressure (Pa)')

# Load the data
data = pd.read_csv('data.csv')
temperature = data.iloc[:, 0]  # Assuming first column is temperature
pressure = data.iloc[:, 1]      # Assuming second column is pressure

# Plot the Phase Boundaries
ax.plot(temperature, pressure, label='Solid-Liquid-Gas Boundary', color='blue')
ax.plot(data.iloc[:, 2], data.iloc[:, 3], label='Solid-Liquid Boundary', color='orange')  # Adjust columns as needed

# Mark Special Points
ax.plot(273.16 - 273.15, 611.657, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')
ax.plot(647.396 - 273.15, 22.064e6, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Draw Freezing and Boiling Points
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')
ax.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Color the Phases
ax.fill_betweenx(pressure, -50, 0, where=(pressure < 611.657), color='lightblue', alpha=0.5, label='Solid Phase')
ax.fill_betweenx(pressure, 0, 100, where=(pressure < 611.657), color='lightgreen', alpha=0.5, label='Liquid Phase')
ax.fill_betweenx(pressure, 100, 200, where=(pressure > 611.657), color='lightyellow', alpha=0.5, label='Gas Phase')

# Set Logarithmic Scale for Pressure
ax.set_yscale('log')

# Add a Grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a Legend
ax.legend()

# Save the Plot
plt.title('Phase Diagram of Water')
plt.savefig('novice_final.png')
plt.show()