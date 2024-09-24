import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("data.csv")

# Calculate average sales for each brand
averages = data.drop(columns=['Quarter']).mean()

# Set the color palette
palette = sns.color_palette("husl", len(data['Quarter'].unique()))

# Create a box plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=data.drop(columns=['Quarter']), palette=palette)

# Overlay individual data points
sns.stripplot(data=data.drop(columns=['Quarter']), color='black', alpha=0.5, size=3)

# Plot average sales
plt.plot(averages.index, averages.values, marker='o', color='red', label='Average Sales')

# Create a color mapping for quarters
quarter_colors = {quarter: color for quarter, color in zip(data['Quarter'].unique(), palette)}

# Add legend and labels
plt.title("Sales Distribution of Mobile Phone Brands by Quarter")
plt.xlabel("Brand")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.legend(title='Quarter', loc='upper right')

# Save the plot
plt.savefig("novice_final.png", bbox_inches='tight')
plt.show()