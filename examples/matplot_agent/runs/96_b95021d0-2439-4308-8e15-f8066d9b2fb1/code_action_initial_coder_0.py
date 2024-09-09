import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data.csv')

# Melt the DataFrame to long format
data_melted = data.melt(id_vars='Quarter', var_name='Brand', value_name='Sales')

# Set the color palette
palette = sns.color_palette("husl", len(data['Quarter'].unique()))

# Create the box plots
plt.figure(figsize=(12, 6))
box_plot = sns.boxplot(x='Brand', y='Sales', hue='Quarter', data=data_melted, palette=palette, dodge=True)

# Overlay individual data points
sns.stripplot(x='Brand', y='Sales', hue='Quarter', data=data_melted, palette=palette, dodge=True, alpha=0.5, marker='o', ax=box_plot)

# Calculate average sales for each brand
average_sales = data.mean().drop('Quarter').reset_index()
average_sales.columns = ['Brand', 'Average Sales']

# Add average sales line
for index, row in average_sales.iterrows():
    plt.text(index, row['Average Sales'], f"{row['Average Sales']:.1f}", color='black', ha='center')
    plt.plot([index, index], [0, row['Average Sales']], color='red', linestyle='--', linewidth=1)

# Customize the plot
plt.title('Sales Distribution of Mobile Phone Brands by Quarter')
plt.ylabel('Sales')
plt.xlabel('Brand')
plt.legend(title='Quarter', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot to a file
plt.savefig('plot.png')

# Show the plot
plt.show()