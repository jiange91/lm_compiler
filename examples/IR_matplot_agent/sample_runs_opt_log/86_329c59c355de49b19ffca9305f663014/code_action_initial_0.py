import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the data from the CSV file
data = pd.read_csv('data.csv')

# Step 2: Set the index to the 'Version' column
data.set_index('Version', inplace=True)

# Step 3: Calculate the market share of other operating systems for each year
data.loc['Other'] = 100 - data.sum()

# Step 4: Define colors for each Windows version
colors = {
    'WinXP': ['#FF9999', '#FF6666', '#FF3333', '#FF0000', '#CC0000'],
    'Win7': ['#99CCFF', '#66B2FF', '#3399FF', '#007FFF', '#005EB8'],
    'Win8.1': ['#FFCC99', '#FFB266', '#FFA64D', '#FF9900', '#CC7A00'],
    'Win10': ['#99FF99', '#66FF66', '#33FF33', '#00FF00', '#00CC00'],
    'Other': ['#FFFFFF'] * 5  # White for other OS
}

# Step 5: Create a figure
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))

# Step 6: Create the doughnut chart
years = data.columns
for i, year in enumerate(years):
    # Get the market share for the current year
    market_share = data[year]
    
    # Create a wedge for each version
    wedges, texts = ax.pie(market_share, labels=market_share.index, 
                           colors=[colors[version][i] for version in market_share.index],
                           startangle=90, counterclock=False, 
                           wedgeprops=dict(width=0.3, edgecolor='w'))
    
    # Add a white section for 'Other' OS
    other_share = market_share['Other']
    wedges, texts = ax.pie([other_share, 100 - other_share], 
                           colors=['#FFFFFF', '#FFFFFF'],
                           startangle=90, counterclock=False, 
                           radius=0.3 + 0.1 * i, 
                           wedgeprops=dict(width=0.1, edgecolor='w'))

    # Annotate each segment with its market share percentage
    for j, wedge in enumerate(wedges):
        angle = (wedge.theta1 + wedge.theta2) / 2
        x = wedge.r * 0.5 * np.cos(np.deg2rad(angle))
        y = wedge.r * 0.5 * np.sin(np.deg2rad(angle))
        ax.text(x, y, f'{market_share[j]:.1f}%', ha='center', va='center', color='black')

# Step 7: Add a legend
ax.legend(wedges, market_share.index, title="Windows Versions", loc="center", fontsize='small')

# Step 8: Title of the chart
plt.title("Desktop Windows Version Market Share Worldwide", fontsize=16)

# Step 9: Save the plot to a file
plt.savefig('novice.png')

# Step 10: Show the plot
plt.show()