import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv("data.csv")

# Prepare the data
phase_boundary = data.iloc[:, [0, 1]]  # Solid-liquid-gas boundary
solid_liquid_boundary = data.iloc[:, [2, 3]]  # Solid-liquid boundary

# Set up the plot
plt.figure(figsize=(10, 6))
plt.title('Phase Diagram of Water')
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (Pa)')

# Convert temperature to Kelvin and pressure to bars
temperature_kelvin = phase_boundary[0] + 273.15
pressure_bars = phase_boundary[1] / 1e5

# Plot the phase boundaries
plt.plot(temperature_kelvin, pressure_bars, label='Solid-Liquid-Gas Boundary', color='blue')
plt.plot(solid_liquid_boundary[2] + 273.15, solid_liquid_boundary[3] / 1e5, label='Solid-Liquid Boundary', color='green')

# Mark special points
plt.plot(273.16, 611.657 / 1e5, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')
plt.plot(647.396, 22.064 * 1e5 / 1e5, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Draw freezing and boiling points
plt.axvline(x=0 + 273.15, color='red', linestyle='--', label='Freezing Point (0 °C)')
plt.axvline(x=100 + 273.15, color='red', linestyle='--', label='Boiling Point (100 °C)')

# Color the phases
plt.fill_betweenx(y=[0, 22.064], x1=0 + 273.15, x2=647.396, color='lightblue', alpha=0.5, label='Gas Phase')
plt.fill_betweenx(y=[611.657 / 1e5, 22.064], x1=0 + 273.15, x2=100 + 273.15, color='lightgreen', alpha=0.5, label='Liquid Phase')
plt.fill_betweenx(y=[0, 611.657 / 1e5], x1=0 + 273.15, x2=0 + 273.15, color='lightgray', alpha=0.5, label='Solid Phase')

# Set logarithmic scale for pressure
plt.yscale('log')

# Add grid and legend
plt.grid(True)
plt.legend()

# Save the plot to a PNG file
plt.savefig("plot.png")

# Show the plot
plt.show()