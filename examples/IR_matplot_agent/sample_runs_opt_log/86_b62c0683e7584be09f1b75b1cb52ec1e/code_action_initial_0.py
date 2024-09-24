import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
data = pd.read_csv('data.csv')

# Set the index to the 'Version' column
data.set_index('Version', inplace=True)

# Calculate the market share of other operating systems
data.loc['Other'] = 100 - data.sum()

# Define colors for each Windows version
colors = {
    'WinXP': ['#FF9999', '#FF6666', '#FF3333', '#FF0000', '#CC0000'],
    'Win7': ['#99CCFF', '#66A3FF', '#3385FF', '#0066FF', '#0052CC'],
    'Win8.1': ['#FFCC99', '#FFB266', '#FFA64D', '#FF9900', '#CC7A00'],
    'Win10': ['#99FF99', '#66FF66', '#33FF33', '#00FF00', '#00CC00'],
    'Other': ['#FFFFFF'] * 5  # White for other OS
}

# Create a figure
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))

# Create a list to hold the start angle for each ring
start_angle = 90  # Start from the top

# Loop through each year to create concentric rings
for i, year in enumerate(data.columns):
    sizes = data[year].values
    # Create a wedge for each version
    wedges, texts = ax.pie(sizes, startangle=start_angle, colors=[colors[version][i] for version in data.index],
                           radius=1 - (i * 0.1), wedgeprops=dict(width=0.1, edgecolor='w'), labels=None)
    
    # Annotate each segment with its respective market share percentage
    for j, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = wedge.r * 0.5 * np.cos(np.radians(angle))
        y = wedge.r * 0.5 * np.sin(np.radians(angle))
        ax.text(x, y, f'{sizes[j]:.1f}%', ha='center', va='center', color='black')

    # Add the white section for 'Other' OS
    other_size = data.loc['Other', year]
    ax.pie([other_size], startangle=start_angle, colors=['#FFFFFF'], radius=1 - (i * 0.1), 
            wedgeprops=dict(width=0.1, edgecolor='w'), labels=[year])
    
    start_angle += 360 / len(sizes)  # Adjust the start angle for the next ring

# Add a legend in the center
ax.legend(data.index, title="Windows Versions", loc="center", bbox_to_anchor=(0.5, 0.5))

# Set the title
plt.title("Desktop Windows Version Market Share Worldwide", fontsize=16)

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()