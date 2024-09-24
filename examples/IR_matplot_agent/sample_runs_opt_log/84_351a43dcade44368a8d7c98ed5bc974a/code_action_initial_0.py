import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from 'data.csv'
data = pd.read_csv('data.csv', header=None)

# Extract relevant columns
temperature_phase = data.iloc[:, 0]
pressure_phase = data.iloc[:, 1]
temperature_solid_liquid = data.iloc[:, 2]
pressure_solid_liquid = data.iloc[:, 3]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set labels for the axes
ax.set_xlabel('Temperature (°C)', fontsize=12)
ax.set_ylabel('Pressure (Pa)', fontsize=12)
ax.set_title('Phase Diagram of Water', fontsize=14)

# Set the y-axis to logarithmic scale
ax.set_yscale('log')

# Plot the phase boundaries
ax.plot(temperature_phase, pressure_phase, label='Phase Boundary (Solid-Liquid-Gas)', color='blue')
ax.plot(temperature_solid_liquid, pressure_solid_liquid, label='Phase Boundary (Solid-Liquid)', color='green')

# Mark the triple point
triple_point = (273.16, 611.657)
ax.plot(triple_point[0] - 273.15, triple_point[1], 'ro', label='Triple Point (273.16 K, 611.657 Pa)')

# Mark the critical point
critical_point = (647.396, 22064000)
ax.plot(critical_point[0] - 273.15, critical_point[1], 'go', label='Critical Point (647.396 K, 22.064 MPa)')

# Freezing point at 0°C
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')

# Boiling point at 100°C
ax.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Fill the regions
ax.fill_betweenx(y=[1e2, 1e7], x1=-100, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')
ax.fill_betweenx(y=[1e2, 1e7], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')
ax.fill_betweenx(y=[1e2, 1e7], x1=100, x2=300, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend
ax.legend()

# Save the plot to a PNG file
plt.savefig("novice.png")

# Show the plot
plt.show()