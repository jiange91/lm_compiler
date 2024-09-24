import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from CSV
data = pd.read_csv('data.csv')

# Convert temperature from Celsius to Kelvin
data['Temperature_K'] = data.iloc[:, 0] + 273.15  # Column 1
data['Pressure_Pa'] = data.iloc[:, 1]              # Column 2
solid_liquid_temp = data.iloc[:, 2] + 273.15       # Column 3
solid_liquid_pressure = data.iloc[:, 3]             # Column 4

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set the x and y axis labels
ax.set_xlabel('Temperature (°C)', fontsize=12)
ax.set_ylabel('Pressure (Pa)', fontsize=12)

# Set the x-axis limits
ax.set_xlim(-50, 200)  # Adjust as necessary for your data
# Set the y-axis limits
ax.set_ylim(0, 25e6)  # 0 to 25 MPa

# Set the y-axis to logarithmic scale
ax.set_yscale('log')

# Add grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot the solid-liquid-gas boundary
ax.plot(data['Temperature_K'], data['Pressure_Pa'], label='Solid-Liquid-Gas Boundary', color='blue')

# Plot the solid-liquid boundary
ax.plot(solid_liquid_temp, solid_liquid_pressure, label='Solid-Liquid Boundary', color='green')

# Mark the triple point
triple_point = (273.16, 611.657)  # (Temperature in K, Pressure in Pa)
ax.plot(triple_point[0] - 273.15, triple_point[1], 'ro', label='Triple Point')

# Mark the critical point
critical_point = (647.396, 22.064e6)  # (Temperature in K, Pressure in Pa)
ax.plot(critical_point[0] - 273.15, critical_point[1], 'ro', label='Critical Point')

# Freezing point at 0°C
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')

# Boiling point at 100°C
ax.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Fill the solid phase
ax.fill_betweenx(y=[0, 611.657], x1=-50, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')

# Fill the liquid phase
ax.fill_betweenx(y=[611.657, 22.064e6], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')

# Fill the gas phase
ax.fill_betweenx(y=[22.064e6, 25e6], x1=100, x2=200, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add legend
ax.legend()

# Save the plot to a PNG file
plt.title('Phase Diagram of Water', fontsize=14)
plt.savefig("novice.png")

# Show the plot
plt.show()