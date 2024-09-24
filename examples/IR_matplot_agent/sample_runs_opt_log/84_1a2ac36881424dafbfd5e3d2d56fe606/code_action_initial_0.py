import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the data from the CSV file
data = pd.read_csv('data.csv')

# Step 2: Extract the relevant columns
temp_gas_liquid = data.iloc[:, 0]  # Column 1 for gas-liquid boundary
pressure_gas_liquid = data.iloc[:, 1]  # Column 2 for gas-liquid boundary
temp_solid_liquid = data.iloc[:, 2]  # Column 3 for solid-liquid boundary
pressure_solid_liquid = data.iloc[:, 3]  # Column 4 for solid-liquid boundary

# Step 3: Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set labels for the axes
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Pressure (Pa)')

# Set the limits for the axes
ax.set_xlim(-50, 100)  # Adjust as necessary
ax.set_ylim(1e2, 1e7)  # Logarithmic scale for pressure

# Step 4: Plot the phase boundaries
ax.plot(temp_gas_liquid, pressure_gas_liquid, label='Gas-Liquid Boundary', color='blue')
ax.plot(temp_solid_liquid, pressure_solid_liquid, label='Solid-Liquid Boundary', color='green')

# Step 5: Mark special points
triple_point = (273.16, 611.657)  # (Temperature in K, Pressure in Pa)
ax.plot(triple_point[0] - 273.15, triple_point[1], 'ro', label='Triple Point')

critical_point = (647.396, 22.064e6)  # (Temperature in K, Pressure in Pa)
ax.plot(critical_point[0] - 273.15, critical_point[1], 'go', label='Critical Point')

# Step 6: Draw freezing and boiling points
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')
ax.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Step 7: Color the phases
ax.fill_betweenx(y=[1e2, 611.657], x1=-50, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')
ax.fill_betweenx(y=[611.657, 22.064e6], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')
ax.fill_betweenx(y=[22.064e6, 1e7], x1=100, x2=150, color='lightyellow', alpha=0.5, label='Gas Phase')

# Step 8: Set logarithmic scale for pressure
ax.set_yscale('log')

# Step 9: Add grid and legend
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()

# Step 10: Save the plot to a file
plt.title('Phase Diagram of Water')
plt.savefig('novice.png')

# Show the plot
plt.show()