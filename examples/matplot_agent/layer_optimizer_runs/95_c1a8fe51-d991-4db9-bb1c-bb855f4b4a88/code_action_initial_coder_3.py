import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Data
data = pd.read_csv('data.csv')

# Step 2: Prepare the Data
# Melt the DataFrame to long format for easier plotting with seaborn
data_melted = data.melt(id_vars=['Quarter'], var_name='Brand', value_name='Sales')

# Calculate the average sales for each brand by quarter
average_sales = data_melted.groupby(['Quarter', 'Brand'])['Sales'].mean().reset_index()
average_sales.rename(columns={'Sales': 'Average Sales'}, inplace=True)

# Step 3: Create the Plot
plt.figure(figsize=(14, 8))

# Create a box plot for each brand
sns.boxplot(x='Brand', y='Sales', data=data_melted, palette='Set2')

# Overlay individual data points
sns.stripplot(x='Brand', y='Sales', data=data_melted, color='black', alpha=0.5, jitter=True)

# Plot the average sales as a line chart
for brand in data['Brand'].unique():
    brand_data = average_sales[average_sales['Brand'] == brand]
    plt.plot(brand_data['Quarter'], brand_data['Average Sales'], marker='o', linestyle='-', label=f'Average {brand}')

# Step 4: Customize the Plot
plt.legend(title='Average Sales')
plt.title('Sales Data for Mobile Phone Brands')
plt.xlabel('Brand')
plt.ylabel('Sales')

# Save the plot to a PNG file
plt.savefig('plot.png')

# Show the plot
plt.show()