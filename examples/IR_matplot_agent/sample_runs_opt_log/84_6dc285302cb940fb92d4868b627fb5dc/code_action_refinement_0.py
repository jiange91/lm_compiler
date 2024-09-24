import matplotlib.pyplot as plt
import numpy as np

# Load data from CSV
data = np.loadtxt('data.csv', delimiter=',')
temp = data[:, 0]  # Assuming column 1 is temperature
pressure = data[:, 1]  # Assuming column 2 is pressure
solid_liquid_temp = data[:, 2]  # Assuming column 3 is temperature for solid-liquid boundary
solid_liquid_pressure = data[:, 3]  # Assuming column 4 is pressure for solid-liquid boundary

# Set up the plot
plt.figure(figsize=(10, 6))
plt.xlabel('Temperature (°C / K)')
plt.ylabel('Pressure (Pa / bars / mbar)')
plt.yscale('log')

# Plot the phase boundaries
plt.plot(temp, pressure, label='Solid-Liquid-Gas Boundary', color='blue')
plt.plot(solid_liquid_temp, solid_liquid_pressure, label='Solid-Liquid Boundary', color='green')

# Mark special points
plt.plot(273.16, 611.657, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')
plt.plot(647.396, 22.064 * 1e6, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Draw freezing and boiling points
plt.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0 °C)')
plt.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100 °C)')

# Color the phases
plt.fill_betweenx(y=[0, 611.657], x1=-10, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')
plt.fill_betweenx(y=[611.657, 22.064 * 1e6], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')
plt.fill_betweenx(y=[22.064 * 1e6, 1e9], x1=100, x2=800, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add grid and legend
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Title and save the plot
plt.title('Phase Diagram of Water')
plt.savefig('novice_final.png')
plt.show()