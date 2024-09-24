# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the Data
data = pd.read_csv('data.csv')

# Step 3: Data Preparation
# Extracting data for phase boundaries
temp_solid_liquid_gas = data.iloc[:, 0]  # Column 1: Temperature for solid-liquid-gas boundary
pressure_solid_liquid_gas = data.iloc[:, 1]  # Column 2: Pressure for solid-liquid-gas boundary
temp_solid_liquid = data.iloc[:, 2]  # Column 3: Temperature for solid-liquid boundary
pressure_solid_liquid = data.iloc[:, 3]  # Column 4: Pressure for solid-liquid boundary

# Step 4: Set Up the Phase Diagram
fig, ax = plt.subplots(figsize=(10, 8))

# Set labels and limits
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Pressure (Pa)')
ax.set_title('Phase Diagram of Water')

# Set the x-axis limits
ax.set_xlim(-50, 200)  # Adjust as necessary for your data
# Set the y-axis limits
ax.set_ylim(1e2, 1e7)  # Logarithmic scale for pressure

# Step 5: Plot the Phase Boundaries
# Plot the solid-liquid-gas boundary
ax.plot(temp_solid_liquid_gas, pressure_solid_liquid_gas, label='Solid-Liquid-Gas Boundary', color='blue')

# Plot the solid-liquid boundary
ax.plot(temp_solid_liquid, pressure_solid_liquid, label='Solid-Liquid Boundary', color='green')

# Step 6: Mark Special Points
# Mark the triple point
triple_point = (0, 611.657)  # (Temperature in °C, Pressure in Pa)
ax.plot(triple_point[0], triple_point[1], 'ro', label='Triple Point (273.16 K, 611.657 Pa)')

# Mark the critical point
critical_point = (374.0, 22.064e6)  # (Temperature in °C, Pressure in Pa)
ax.plot(critical_point[0], critical_point[1], 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Step 7: Draw Freezing and Boiling Points
# Freezing point at 0°C
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')

# Boiling point at 100°C
ax.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Step 8: Color the Phases
# Fill the solid phase
ax.fill_betweenx(y=[1e2, 611.657], x1=-50, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')

# Fill the liquid phase
ax.fill_betweenx(y=[611.657, 22.064e6], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')

# Fill the gas phase
ax.fill_betweenx(y=[22.064e6, 1e7], x1=100, x2=200, color='lightyellow', alpha=0.5, label='Gas Phase')

# Step 9: Add a Grid and Set Logarithmic Scale
# Add grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Set y-axis to logarithmic scale
ax.set_yscale('log')

# Step 10: Add Legend and Show the Plot
# Add legend
ax.legend()

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()