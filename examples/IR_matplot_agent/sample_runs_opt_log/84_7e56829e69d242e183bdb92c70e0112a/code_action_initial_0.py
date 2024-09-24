# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the Data
data = pd.read_csv('data.csv')

# Step 3: Prepare the Data
# Extracting data for phase boundaries
temp_gas_liquid = data.iloc[:, 0]  # Column 1 for gas-liquid boundary
pressure_gas_liquid = data.iloc[:, 1]  # Column 2 for gas-liquid boundary
temp_solid_liquid = data.iloc[:, 2]  # Column 3 for solid-liquid boundary
pressure_solid_liquid = data.iloc[:, 3]  # Column 4 for solid-liquid boundary

# Step 4: Set Up the Plot
plt.figure(figsize=(10, 6))
plt.title('Phase Diagram of Water')
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (Pa)')
plt.xlim(-50, 200)  # Set limits for temperature
plt.ylim(1e2, 1e7)  # Set limits for pressure
plt.yscale('log')  # Set y-axis to logarithmic scale

# Step 5: Plot the Phase Boundaries
plt.plot(temp_gas_liquid, pressure_gas_liquid, label='Gas-Liquid Boundary', color='blue')
plt.plot(temp_solid_liquid, pressure_solid_liquid, label='Solid-Liquid Boundary', color='green')

# Step 6: Mark Special Points
plt.plot(0.01, 611.657, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')
plt.plot(374.0, 22.064e6, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Step 7: Draw Freezing and Boiling Points
plt.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')
plt.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Step 8: Color the Regions
plt.fill_betweenx(y=[1e2, 611.657], x1=-50, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')
plt.fill_betweenx(y=[611.657, 22.064e6], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')
plt.fill_betweenx(y=[22.064e6, 1e7], x1=100, x2=200, color='lightyellow', alpha=0.5, label='Gas Phase')

# Step 9: Add a Grid and Legend
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Step 10: Save the Plot to a File
plt.savefig('novice.png')

# Step 11: Show the Plot
plt.show()