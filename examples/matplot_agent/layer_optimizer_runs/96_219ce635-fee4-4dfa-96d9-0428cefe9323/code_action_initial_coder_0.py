import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
data = pd.read_csv('data.csv')

# Check the first few rows of the DataFrame to understand its structure
print(data.head())

# Melt the DataFrame to have a long-form DataFrame suitable for seaborn
data_melted = pd.melt(data, id_vars=['Quarter'], var_name='Brand', value_name='Sales')

# Set the color palette
palette = sns.color_palette("husl", len(data['Quarter'].unique()))

# Create the box plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Brand', y='Sales', data=data_melted, palette=palette)

# Overlay individual data points
sns.stripplot(x='Brand', y='Sales', data=data_melted, color='black', alpha=0.5, jitter=True)

# Calculate the average sales for each brand
average_sales = data_melted.groupby('Brand')['Sales'].mean().reset_index()

# Plot the average sales line
plt.plot(average_sales['Brand'], average_sales['Sales'], marker='o', color='red', linestyle='-', linewidth=2, label='Average Sales')

# Add a legend
plt.legend()

# Add titles and labels
plt.title('Sales Data for Mobile Phone Brands')
plt.xlabel('Brand')
plt.ylabel('Sales')

# Save the plot to a file
plot_file_name = 'plot.png'
plt.savefig(plot_file_name)

# Show the plot
plt.show()