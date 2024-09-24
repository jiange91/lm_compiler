import json
import plotly.graph_objects as go

# Step 1: Open the data file
with open('data.json', 'r') as file:
    data = json.load(file)

# Step 2: Extracting the relevant data
sankey_data = data['data'][0]
nodes = sankey_data['node']
links = sankey_data['link']

# Step 3: Prepare node labels and colors
node_labels = nodes['label']
node_colors = nodes['color']

# Step 4: Prepare link source, target, value, and color
link_source = links['source']
link_target = links['target']
link_values = links['value']
link_colors = links['color']

# Step 5: Create the Sankey diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=node_labels,
        color=node_colors
    ),
    link=dict(
        source=link_source,
        target=link_target,
        value=link_values,
        color=link_colors
    )
))

# Step 6: Update layout settings
fig.update_layout(
    title_text="Energy Flow Diagram",
    font_size=10,
    height=600,
    width=800
)

# Step 7: Save the diagram to a file
fig.write_image("novice.png")

# Show the diagram
fig.show()