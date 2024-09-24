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
sns.boxplot(x='Quarter', y='Sales', hue='Brand', data=melted_data, palette=palette, dodge=True)
sns.stripplot(x='Quarter', y='Sales', hue='Brand', data=melted_data, palette=palette, dodge=True, marker='o', alpha=0.5)

# Calculate average sales for each brand and quarter
average_sales = melted_data.groupby(['Quarter', 'Brand'])['Sales'].mean().reset_index()

# Plot average sales
for brand in average_sales['Brand'].unique():
    brand_data = average_sales[average_sales['Brand'] == brand]
    plt.plot(brand_data['Quarter'], brand_data['Sales'], marker='o', label=f'Avg {brand}')

# Customize the plot
plt.title('Mobile Phone Sales Distribution by Brand and Quarter')
plt.xlabel('Quarter')
plt.ylabel('Sales')
plt.legend(title='Brand', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()