import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data.csv')

# Convert 'Quarter' to a categorical type
data['Quarter'] = pd.Categorical(data['Quarter'], ordered=True)

# Melt the DataFrame to long format
melted_data = data.melt(id_vars='Quarter', var_name='Brand', value_name='Sales')

# Set the color palette
palette = sns.color_palette("husl", len(data['Quarter'].unique()))

# Create the box plot
plt.figure(figsize=(12, 6))
box_plot = sns.boxplot(x='Brand', y='Sales', hue='Quarter', data=melted_data, palette=palette)

# Overlay individual data points
sns.stripplot(x='Brand', y='Sales', hue='Quarter', data=melted_data, dodge=True, color='black', alpha=0.5, size=3)

# Calculate average sales
average_sales = melted_data.groupby(['Brand', 'Quarter'])['Sales'].mean().reset_index()

# Plot average sales
for brand in average_sales['Brand'].unique():
    brand_data = average_sales[average_sales['Brand'] == brand]
    plt.plot(brand_data['Brand'], brand_data['Sales'], marker='o', label=f'Avg {brand}')

# Customize the plot
plt.title('Sales Distribution of Mobile Phone Brands by Quarter')
plt.xlabel('Brand')
plt.ylabel('Sales')
plt.legend(title='Quarter', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot to a PNG file
plt.savefig('plot.png')

# Show the plot
plt.show()