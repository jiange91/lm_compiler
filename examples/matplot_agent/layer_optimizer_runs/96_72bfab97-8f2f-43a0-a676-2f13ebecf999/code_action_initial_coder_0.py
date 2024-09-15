import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data.csv')

# Melt the DataFrame to long format
melted_data = data.melt(id_vars='Quarter', var_name='Brand', value_name='Sales')

# Calculate average sales for each brand
average_sales = melted_data.groupby('Brand')['Sales'].mean().reset_index()

# Set the color palette
palette = sns.color_palette("husl", len(data['Quarter'].unique()))

# Create the box plots
plt.figure(figsize=(12, 6))
box_plot = sns.boxplot(x='Brand', y='Sales', hue='Quarter', data=melted_data, palette=palette, dodge=True)

# Overlay individual data points
sns.stripplot(x='Brand', y='Sales', hue='Quarter', data=melted_data, palette=palette, dodge=True, alpha=0.5, size=5)

# Plot average sales
for index, row in average_sales.iterrows():
    plt.plot([index, index], [0, row['Sales']], color='black', linestyle='--', linewidth=2)
    plt.text(index, row['Sales'], f'{row["Sales"]:.1f}', color='black', ha='center', va='bottom')

# Customize the plot
plt.title('Sales Distribution of Mobile Phone Brands by Quarter')
plt.xlabel('Brand')
plt.ylabel('Sales')
plt.legend(title='Quarter', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot to a file
plt.savefig('plot.png')

# Show the plot
plt.show()