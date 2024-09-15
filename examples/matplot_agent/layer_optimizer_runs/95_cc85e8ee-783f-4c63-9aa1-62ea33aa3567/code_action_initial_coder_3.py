import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Data
data = pd.read_csv('data.csv')

# Ensure the column names are stripped of any leading/trailing whitespace
data.columns = data.columns.str.strip()

# Check if the required columns are present in the DataFrame
required_columns = ['Quarter', 'Samsung', 'Nokia/Microsoft', 'Apple', 'LG', 'ZTE', 'Huawei']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing from the DataFrame: {missing_columns}")

# Step 2: Prepare the Data
brands = ["Samsung", "Nokia/Microsoft", "Apple", "LG", "ZTE", "Huawei"]
data_long = pd.melt(data, id_vars=['Quarter'], value_vars=brands, var_name='Brand', value_name='Sales')

# Convert 'Sales' to numeric, forcing errors to NaN
data_long['Sales'] = pd.to_numeric(data_long['Sales'], errors='coerce')

# Drop rows with NaN values in 'Sales'
data_long = data_long.dropna(subset=['Sales'])

# Calculate the average sales for each brand
average_sales = data_long.groupby('Brand')['Sales'].mean().reset_index()

# Step 3: Create the Box Plots
plt.figure(figsize=(12, 8))
sns.boxplot(x='Brand', y='Sales', data=data_long, palette='Set3')

# Overlay individual sales data points
sns.stripplot(x='Brand', y='Sales', data=data_long, color='black', alpha=0.5, jitter=True)

# Step 4: Plot the Average Sales Line
avg_sales_values = average_sales.set_index('Brand').loc[brands]['Sales'].values
plt.plot(brands, avg_sales_values, color='red', marker='o', linestyle='-', linewidth=2, markersize=8, label='Average Sales')

# Step 5: Customize the Plot
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

# Step 6: Save the Plot
plt.savefig('plot.png')

# Show the plot
plt.show()