# Step 1: Import Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Data
data = pd.read_csv('data.csv')

# Step 3: Prepare the Data
# Extract the sales data for each brand
brands = ["Samsung", "Nokia/Microsoft", "Apple", "LG", "ZTE", "Huawei"]
data_long = pd.melt(data, id_vars=['Quarter'], value_vars=brands, var_name='Brand', value_name='Sales')

# Calculate the average sales for each brand
average_sales = data_long.groupby('Brand')['Sales'].mean().reset_index()

# Step 4: Create the Box Plots
plt.figure(figsize=(12, 8))
sns.boxplot(x='Brand', y='Sales', data=data_long, palette='Set3')

# Overlay individual sales data points
sns.stripplot(x='Brand', y='Sales', data=data_long, color='black', alpha=0.5, jitter=True)

# Step 5: Plot the Average Sales Line
# Extract the average sales values in the same order as the brands
avg_sales_values = average_sales.set_index('Brand').loc[brands]['Sales'].values

# Plot the line connecting the average sales values
plt.plot(brands, avg_sales_values, color='red', marker='o', linestyle='-', linewidth=2, markersize=8, label='Average Sales')

# Step 6: Customize the Plot
# Use a consistent color scheme for the same quarter across different years
quarters = data['Quarter'].unique()
colors = sns.color_palette('husl', len(quarters))

for i, quarter in enumerate(quarters):
    quarter_data = data_long[data_long['Quarter'] == quarter]
    sns.stripplot(x='Brand', y='Sales', data=quarter_data, color=colors[i], alpha=0.5, jitter=True, label=quarter)

# Add a legend to explain the color coding
plt.legend(title='Quarter')

# Add titles and labels
plt.title('Sales Data for Mobile Phone Brands')
plt.xlabel('Brand')
plt.ylabel('Sales')

# Step 7: Save the Plot
plt.savefig('plot.png')

# Show the plot
plt.show()