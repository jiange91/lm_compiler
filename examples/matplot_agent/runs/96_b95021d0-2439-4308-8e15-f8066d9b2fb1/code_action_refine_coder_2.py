import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data.csv')

# Melt the DataFrame to long format
data_melted = data.melt(id_vars='Quarter', var_name='Brand', value_name='Sales')

# Set the color palette
palette = sns.color_palette("Set2", len(data['Quarter'].unique()))

# Create the box plots
plt.figure(figsize=(12, 6))
box_plot = sns.boxplot(x='Brand', y='Sales', hue='Quarter', data=data_melted, palette=palette, dodge=True)

# Overlay individual data points
sns.stripplot(x='Brand', y='Sales', hue='Quarter', data=data_melted, palette=palette, dodge=True, alpha=0.5, marker='o', ax=box_plot)

# Calculate average sales for each brand
average_sales = data_melted.groupby('Brand')['Sales'].mean().reset_index()
average_sales.columns = ['Brand', 'Average Sales']

# Add average sales line
plt.plot(range(len(average_sales)), average_sales['Average Sales'], color='red', marker='o', linestyle='-', linewidth=2)

# Customize the plot
plt.title('Sales Distribution of Mobile Phone Brands by Quarter', fontsize=16)
plt.ylabel('Sales', fontsize=14)
plt.xlabel('Brand', fontsize=14)
plt.legend(title='Quarter', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot to a file
plt.savefig('plot_final.png')

# Show the plot
plt.show()