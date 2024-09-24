import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
data = pd.read_csv('data.csv')
temperature = data.iloc[:, 0]  # Assuming first column is temperature
pressure = data.iloc[:, 1]      # Assuming second column is pressure

# Set up the figure and axes
plt.figure(figsize=(10, 6))
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (Pa)')

# Plot the phase boundaries
plt.plot(temperature, pressure, label='Gas-Liquid Boundary', color='blue')
plt.plot(data.iloc[:, 2], data.iloc[:, 3], label='Solid-Liquid Boundary', color='green')  # Adjust columns as needed

# Mark special points
plt.scatter(273.16, 611.657, color='red', label='Triple Point (273.16 K, 611.657 Pa)')
plt.scatter(647.396, 22.064 * 1e6, color='red', label='Critical Point (647.396 K, 22.064 MPa)')

# Draw freezing and boiling points
plt.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')
plt.axvline(x=100, color='red', linestyle='--', label='Boiling Point (100°C)')

# Color the phases
plt.fill_betweenx(pressure, -40, 0, where=(pressure < 611.657), color='lightblue', alpha=0.5, label='Solid Phase')
plt.fill_betweenx(pressure, 0, 100, where=(pressure < 611.657), color='lightgreen', alpha=0.5, label='Liquid Phase')
plt.fill_betweenx(pressure, 100, 300, where=(pressure > 611.657), color='lightyellow', alpha=0.5, label='Gas Phase')

# Add a logarithmic scale for pressure
plt.yscale('log')

# Add a grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend
plt.legend()

# Save the plot
plt.title('Phase Diagram of Water')
plt.savefig('novice_final.png')
plt.show()