import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data.csv')

# Prepare data for plotting
years = ['2015', '2016', '2017', '2018', '2019']
windows_versions = data.columns[0].tolist()  # Assuming the first column is the Windows versions
market_shares = [data[year].tolist() for year in years]

# Define colors
colors = {
    'WinXP': ['#FF9999', '#FF6666', '#FF3333', '#FF0000', '#CC0000'],
    'Win7': ['#66B3FF', '#3399FF', '#007FFF', '#005EB8', '#003F7F'],
    'Win8.1': ['#99FF99', '#66FF66', '#33FF33', '#00FF00', '#00CC00'],
    'Win10': ['#FFFF99', '#FFFF66', '#FFFF33', '#FFFF00', '#CCCC00'],
}

# Create the doughnut chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))

# Create each ring
for i, year in enumerate(years):
    wedges, texts, autotexts = ax.pie(
        market_shares[i],
        labels=windows_versions,
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors[version][i] for version in windows_versions],
        radius=1 - (i * 0.1),
        wedgeprops=dict(width=0.1, edgecolor='w')
    )
    # Align the white section for "Other"
    other_share = 100 - sum(market_shares[i])
    ax.pie([other_share], radius=1 - (i * 0.1), colors=['white'], startangle=90)

# Add title
plt.title("Desktop Windows Version Market Share Worldwide", fontsize=16)

# Add legend
ax.legend(wedges, windows_versions, title="Windows Versions", loc="center", fontsize=10)

# Save the plot
plt.savefig("novice_final.png")
plt.show()