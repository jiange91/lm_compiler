import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Check the structure of the DataFrame
print(data.head())  # This will help us understand the columns available

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Pressure (Pa)')
ax.set_title('Phase Diagram of Water')

# Convert Pressure to Logarithmic Scale
ax.set_yscale('log')

# Plot the Phase Boundaries
# Assuming columns 0 and 1 are for gas-liquid boundary
# and columns 2 and 3 are for solid-liquid boundary
ax.plot(data.iloc[:, 0], data.iloc[:, 1], label='Gas-Liquid Boundary', color='blue')
ax.plot(data.iloc[:, 2], data.iloc[:, 3], label='Solid-Liquid Boundary', color='green')

# Mark the Special Points
ax.plot(0, 611.657, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')
ax.plot(647.396, 22064000, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Draw Freezing and Boiling Points
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0 °C)')
ax.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100 °C)')

# Color the Phases
ax.fill_betweenx([0, 611.657], -100, 0, color='lightblue', alpha=0.5, label='Solid Phase')
ax.fill_betweenx([611.657, 22064000], 0, 100, color='lightgreen', alpha=0.5, label='Liquid Phase')
ax.fill_betweenx([22064000, 1e9], 100, 600, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add a Grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a Legend
ax.legend()

# Save the Plot
plt.savefig('phase_diagram_water.png')

# Show the Plot
plt.show()