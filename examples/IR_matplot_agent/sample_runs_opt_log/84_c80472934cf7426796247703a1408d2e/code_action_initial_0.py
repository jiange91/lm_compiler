import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
data = pd.read_csv('data.csv')

# Convert temperature from Celsius to Kelvin
data['Temperature_K'] = data.iloc[:, 0] + 273.15  # Column 1
data['Pressure_Pa'] = data.iloc[:, 1]  # Column 2
data['Solid_Liquid_Temperature_K'] = data.iloc[:, 2] + 273.15  # Column 3
data['Solid_Liquid_Pressure_Pa'] = data.iloc[:, 3]  # Column 4

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set labels
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Pressure (Pa)')
ax.set_title('Phase Diagram of Water')

# Plot the solid-liquid and solid-gas boundaries
ax.plot(data['Temperature_K'] - 273.15, data['Pressure_Pa'], label='Solid-Gas Boundary', color='blue')
ax.plot(data['Solid_Liquid_Temperature_K'] - 273.15, data['Solid_Liquid_Pressure_Pa'], label='Solid-Liquid Boundary', color='green')

# Mark the triple point
triple_point = (0, 611.657)  # (Temperature in °C, Pressure in Pa)
ax.plot(triple_point[0], triple_point[1], 'ro', label='Triple Point (273.16 K, 611.657 Pa)')

# Mark the critical point
critical_point = (374, 22064000)  # (Temperature in °C, Pressure in Pa)
ax.plot(critical_point[0], critical_point[1], 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Freezing point at 0°C
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')

# Boiling point at 100°C
ax.axvline(x=100, color='red', linestyle='--', label='Boiling Point (100°C)')

# Fill the regions
ax.fill_betweenx(y=[0, 611.657], x1=-50, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')
ax.fill_betweenx(y=[611.657, 22064000], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')
ax.fill_betweenx(y=[22064000, 1e7], x1=100, x2=374, color='lightyellow', alpha=0.5, label='Gas Phase')

# Set y-axis to logarithmic scale
ax.set_yscale('log')

# Add grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend
ax.legend()

# Save the plot to a PNG file
plt.savefig('novice.png')

# Show the plot
plt.show()