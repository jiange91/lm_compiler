# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the Data
data = pd.read_csv('data.csv')

# Step 3: Inspect the Data
print(data.head())  # Check the first few rows to understand the structure

# Step 4: Data Preparation
# Assuming the CSV has the following columns:
# Column 1: Temperature (in Celsius) - 'Temperature_C'
# Column 2: Pressure (in Pascals) - 'Pressure_Pa'
# Column 3: Solid-Liquid boundary temperature (in Celsius) - 'Solid_Liquid_Temperature_C'
# Column 4: Solid-Liquid boundary pressure (in Pascals) - 'Solid_Liquid_Pressure_Pa'

# Check if the expected columns exist
if 'Temperature_C' not in data.columns or 'Pressure_Pa' not in data.columns:
    raise ValueError("The data must contain 'Temperature_C' and 'Pressure_Pa' columns.")

# Step 5: Set Up the Phase Diagram
fig, ax = plt.subplots(figsize=(10, 8))

# Set the axes labels
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Pressure (Pa)')

# Set the y-axis to logarithmic scale
ax.set_yscale('log')

# Step 6: Plot the Phase Boundaries
# Plot the solid-liquid-gas boundary
ax.plot(data['Temperature_C'], data['Pressure_Pa'], label='Solid-Liquid-Gas Boundary', color='blue')

# Plot the solid-liquid boundary
ax.plot(data['Solid_Liquid_Temperature_C'], data['Solid_Liquid_Pressure_Pa'], label='Solid-Liquid Boundary', color='green')

# Step 7: Mark Special Points
# Mark the triple point
triple_point = (0.01, 611.657)  # (Temperature in °C, Pressure in Pa)
ax.plot(triple_point[0], triple_point[1], 'ro', label='Triple Point (273.16 K, 611.657 Pa)')

# Mark the critical point
critical_point = (374.0, 22064000)  # (Temperature in °C, Pressure in Pa)
ax.plot(critical_point[0], critical_point[1], 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Step 8: Draw Freezing and Boiling Points
# Freezing point at 0 °C
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0 °C)')

# Boiling point at 100 °C
ax.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100 °C)')

# Step 9: Color the Phases
# Fill the solid phase
ax.fill_betweenx(y=[0, 611.657], x1=-50, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')

# Fill the liquid phase
ax.fill_betweenx(y=[611.657, 22064000], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')

# Fill the gas phase
ax.fill_betweenx(y=[22064000, 1e7], x1=100, x2=200, color='lightyellow', alpha=0.5, label='Gas Phase')

# Step 10: Add Grid and Legend
# Add grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend
ax.legend()

# Step 11: Save the Plot to a PNG file
plt.title('Phase Diagram of Water')
plt.savefig("novice.png")

# Show the Plot
plt.show()