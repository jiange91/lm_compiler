# Step 1: Import Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Data
data = pd.read_csv('data.csv')

# Step 3: Prepare the Data
# Melt the DataFrame to long format for easier plotting with seaborn
data_melted = data.melt(id_vars=['Quarter'], var_name='Brand', value_name='Sales')

# Calculate the average sales for each brand
average_sales = data.mean().reset_index()
average_sales.columns = ['Brand', 'Average Sales']

# Step 4: Create the Plot
plt.figure(figsize=(14, 8))

# Create a box plot for each brand
sns.boxplot(x='Brand', y='Sales', data=data_melted, palette='Set2')

# Overlay individual data points
sns.stripplot(x='Brand', y='Sales', data=data_melted, color='black', alpha=0.5, jitter=True)

# Plot the average sales as a line chart
for brand in data.columns[1:]:
    avg_sales = data[brand].mean()
    plt.plot([brand], [avg_sales], marker='o', linestyle='-', label=f'Average {brand}')

# Step 5: Customize the Plot
plt.legend(title='Average Sales')
plt.title('Sales Data for Mobile Phone Brands')
plt.xlabel('Brand')
plt.ylabel('Sales')

# Save the plot to a PNG file
plt.savefig('plot.png')

# Show the plot
plt.show()