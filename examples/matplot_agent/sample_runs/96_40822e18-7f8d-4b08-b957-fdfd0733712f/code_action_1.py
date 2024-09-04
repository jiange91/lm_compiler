import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data.csv')

data['Quarter'] = pd.Categorical(data['Quarter'], ordered=True)
averages = data[['Quarter', 'Samsung', 'Nokia/Microsoft', 'Apple', 'LG', 'ZTE', 'Huawei']].mean(numeric_only=True).reset_index()

melted_data = data.melt(id_vars='Quarter', value_vars=['Samsung', 'Nokia/Microsoft', 'Apple', 'LG', 'ZTE', 'Huawei'], var_name='Brand', value_name='Sales')
plt.figure(figsize=(12, 6))
sns.boxplot(x='Quarter', y='Sales', hue='Brand', data=melted_data, palette='Set2')
sns.stripplot(x='Quarter', y='Sales', hue='Brand', data=melted_data, dodge=True, color='black', alpha=0.5, marker='o')

for brand in ['Samsung', 'Nokia/Microsoft', 'Apple', 'LG', 'ZTE', 'Huawei']:
    plt.plot(averages['Quarter'], averages[brand], marker='o', label=f'Avg {brand}')

plt.title('Sales Distribution of Mobile Phone Brands by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Sales')
plt.legend(title='Brands')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot.png')
plt.show()