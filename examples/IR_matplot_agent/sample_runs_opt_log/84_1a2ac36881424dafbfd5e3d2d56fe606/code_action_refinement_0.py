import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Pressure (Pa)')
ax.set_title('Phase Diagram of Water')

# Convert Pressure to Logarithmic Scale
ax.set_yscale('log')

# Plot the Phase Boundaries
ax.plot(data['Temperature'], data['Pressure'], label='Gas-Liquid Boundary', color='blue')
ax.plot(data['Temperature'], data['Solid-Liquid Boundary'], label='Solid-Liquid Boundary', color='green')

# Mark the Special Points
ax.plot(0, 611.657, 'ro', label='Triple Point')
ax.plot(647.396 - 273.15, 22.064 * 1e6, 'ro', label='Critical Point')  # Convert MPa to Pa

# Draw Freezing and Boiling Points
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')
ax.axvline(x=100, color='red', linestyle='--', label='Boiling Point (100°C)')

# Color the Phases
ax.fill_betweenx(y=[0, 611.657], x1=-40, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')
ax.fill_betweenx(y=[611.657, 22.064 * 1e6], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')
ax.fill_betweenx(y=[22.064 * 1e6, 1e7], x1=100, x2=200, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add a Grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a Legend
ax.legend()

# Save the Plot
plt.savefig('novice_final.png')
plt.show()