import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("data.csv")

# Melt the DataFrame
data_melted = data.melt(id_vars=["Quarter"], var_name="Brand", value_name="Sales")

# Create the Box Plot
plt.figure(figsize=(12, 6))
sns.boxplot(x="Quarter", y="Sales", hue="Brand", data=data_melted, palette="Set2", dodge=True)

# Add Individual Data Points
sns.stripplot(x="Quarter", y="Sales", hue="Brand", data=data_melted, 
              palette="Set2", dodge=True, marker='o', alpha=0.5, jitter=True)

# Calculate Average Sales
averages = data_melted.groupby(['Quarter', 'Brand'])['Sales'].mean().reset_index()

# Plot Average Sales
for brand in averages['Brand'].unique():
    brand_data = averages[averages['Brand'] == brand]
    plt.plot(brand_data['Quarter'], brand_data['Sales'], marker='o', label=f'Avg {brand}')

# Customize the Legend
plt.legend(title='Brand', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add Titles and Labels
plt.title("Mobile Phone Sales Distribution by Brand and Quarter")
plt.xlabel("Quarter")
plt.ylabel("Sales")

# Save the Plot
plt.tight_layout()
plt.savefig("novice_final.png")
plt.show()