import pandas as pd
import plotly.express as px

df = pd.read_csv('data.csv')
df.dropna(subset=['country', 'continent', 'lifeExp', 'pop'], inplace=True)
avg_life_exp = (df['lifeExp'] * df['pop']).sum() / df['pop'].sum()

fig = px.sunburst(
    df,
    path=['continent', 'country'],
    values='pop',
    color='lifeExp',
    color_continuous_scale=px.colors.sequential.RdBu,
    color_continuous_midpoint=avg_life_exp,
    title='Sunburst Plot of Countries by Continent'
)

fig.update_layout(legend_title_text='Life Expectancy')
fig.write_image('plot.png')
fig.show()