# Step 1: Install Required Libraries
# Make sure to install the required libraries if you haven't already
# pip install pandas matplotlib numpy

# Step 2: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 3: Load the Data
data = pd.read_csv('data.csv')

# Step 4: Data Preparation
# Extracting data for phase boundaries
temp_phase_boundary = data.iloc[:, 0]  # Column 1: Temperature for phase boundaries (°C)
pressure_phase_boundary = data.iloc[:, 1]  # Column 2: Pressure for phase boundaries (Pa)
temp_solid_liquid_boundary = data.iloc[:, 2]  # Column 3: Solid-Liquid boundary temperature (°C)
pressure_solid_liquid_boundary = data.iloc[:, 3]  # Column 4: Solid-Liquid boundary pressure (Pa)

# Step 5: Set Up the Phase Diagram
fig, ax = plt.subplots(figsize=(10, 8))

# Set labels and limits
ax.set_xlabel('Temperature (°C)', fontsize=14)
ax.set_ylabel('Pressure (Pa)', fontsize=14)
ax.set_title('Phase Diagram of Water', fontsize=16)

# Set logarithmic scale for pressure
ax.set_yscale('log')

# Set limits for temperature and pressure
ax.set_xlim(-50, 200)  # Adjust as necessary
ax.set_ylim(1e2, 1e7)  # Adjust as necessary

# Step 6: Plot the Phase Boundaries
ax.plot(temp_phase_boundary, pressure_phase_boundary, label='Phase Boundary (Solid-Liquid-Gas)', color='blue')
ax.plot(temp_solid_liquid_boundary, pressure_solid_liquid_boundary, label='Phase Boundary (Solid-Liquid)', color='green')

# Step 7: Mark Special Points
# Mark the triple point
triple_point = (0, 611.657)  # (Temperature in °C, Pressure in Pa)
ax.plot(triple_point[0], triple_point[1], 'ro', label='Triple Point (273.16 K, 611.657 Pa)')

# Mark the critical point
critical_point = (374.0, 22.064e6)  # (Temperature in °C, Pressure in Pa)
ax.plot(critical_point[0], critical_point[1], 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Step 8: Draw Freezing and Boiling Points
# Freezing point at 0°C
ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')

# Boiling point at 100°C
ax.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

# Step 9: Color the Phases
# Fill the regions
ax.fill_betweenx([1e2, 611.657], -50, 0, color='lightblue', alpha=0.5, label='Solid Phase')
ax.fill_betweenx([611.657, 22.064e6], 0, 100, color='lightgreen', alpha=0.5, label='Liquid Phase')
ax.fill_betweenx([22.064e6, 1e7], 100, 200, color='lightyellow', alpha=0.5, label='Gas Phase')

# Step 10: Add Grid and Legend
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()

# Step 11: Save and Show the Plot
plt.savefig('novice.png')  # Save the plot as a PNG file
plt.show()  # Display the plot