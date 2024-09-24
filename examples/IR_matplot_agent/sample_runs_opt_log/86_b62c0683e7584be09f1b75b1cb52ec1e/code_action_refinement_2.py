import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")

# Prepare data
years = data.columns[1:]  # Assuming first column is OS versions
os_versions = data.iloc[:, 0]
market_shares = data.iloc[:, 1:].values

# Calculate 'Other' market share
other_shares = 100 - market_shares.sum(axis=1)
other_shares[other_shares < 0] = 0  # Ensure no negative values

# Create a doughnut chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))

# Define colors for each Windows version
colors = {
    'WinXP': ['#FF9999', '#FF6666', '#FF3333', '#FF0000', '#CC0000'],
    'Win7': ['#66B3FF', '#3399FF', '#007FFF', '#005EB8', '#004080'],
    'Win8.1': ['#99FF99', '#66FF66', '#33FF33', '#00FF00', '#00CC00'],
    'Win10': ['#FFFF99', '#FFFF66', '#FFFF33', '#FFFF00', '#CCCC00'],
}

# Create rings for each year
for i, year in enumerate(years):
    # Create a ring for each year
    sizes = market_shares[:, i].tolist() + [other_shares[i]]
    explode = [0.1] * len(sizes)  # Slightly explode each segment
    start_angle = 90  # Start from the top
    ax.pie(sizes, explode=explode, labels=os_versions.tolist() + ['Other'], 
           colors=[colors['WinXP'][i], colors['Win7'][i], colors['Win8.1'][i], colors['Win10'][i], 'white'],
           autopct='%1.1f%%', startangle=start_angle, counterclock=False)

# Add title
plt.title("Desktop Windows Version Market Share Worldwide", fontsize=16)

# Add legend
ax.legend(os_versions.tolist() + ['Other'], loc='center', fontsize=12)

# Save the plot
plt.savefig("novice_final.png", bbox_inches='tight')
plt.show()