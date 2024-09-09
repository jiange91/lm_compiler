from compiler.utils import load_api_key, get_bill
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


import json

load_api_key('secrets.toml')
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)

human_prompt = "Summarize our conversation so far in {word_count} words."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

print(human_message_template)

msgs = human_message_template.format_messages(**{"word_count":10})
print(human_message_template)
print(msgs)

msgs = [
    SystemMessage(content="\nYou are a cutting-edge super capable code generation LLM. You will be given a natural language query, generate a runnable python code to satisfy all the requirements in the query. You can use any python library you want. \n\nIf the query requires data manipulation from a csv file, process the data from the csv file and draw the plot in one piece of code.\n\nIn your code, when you complete a plot, remember to save it to a png file with given the 'plot_file_name'.\n"), 
    HumanMessage(content=['- expended_query:\nTo create a combination chart with box plots and average sales lines from the "data.csv" dataset, follow these detailed step-by-step instructions. We will use the `pandas`, `matplotlib`, and `seaborn` libraries in Python. \n\n### Step 1: Install Required Libraries\nMake sure you have the required libraries installed. You can install them using pip if you haven\'t done so already:\n\n```bash\npip install pandas matplotlib seaborn\n```\n\n### Step 2: Import Libraries\nStart your Python script or Jupyter Notebook by importing the necessary libraries:\n\n```python\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n```\n\n### Step 3: Load the Data\nLoad the dataset from the "data.csv" file using `pandas`. Ensure that the file is in the same directory as your script or provide the full path.\n\n```python\n# Load the dataset\ndata = pd.read_csv(\'data.csv\')\n```\n\n### Step 4: Prepare the Data\nTo create box plots and calculate averages, we need to reshape the data. We will use the `melt` function to convert the wide format into a long format.\n\n```python\n# Reshape the data to long format\ndata_long = data.melt(id_vars=\'Quarter\', var_name=\'Brand\', value_name=\'Sales\')\n```\n\n### Step 5: Calculate Average Sales\nCalculate the average sales for each brand and store it in a new DataFrame.\n\n```python\n# Calculate average sales for each brand\naverage_sales = data_long.groupby(\'Brand\')[\'Sales\'].mean().reset_index()\n```\n\n### Step 6: Create the Box Plots\nUse `seaborn` to create box plots for each brand. We will also overlay the individual sales data points using the `stripplot` function.\n\n```python\n# Set the color palette\npalette = sns.color_palette("husl", len(data[\'Quarter\'].unique()))\n\n# Create the box plots\nplt.figure(figsize=(12, 6))\nsns.boxplot(x=\'Brand\', y=\'Sales\', data=data_long, palette=palette)\n\n# Overlay individual sales data points\nsns.stripplot(x=\'Brand\', y=\'Sales\', data=data_long, color=\'black\', alpha=0.5, jitter=True)\n```\n\n### Step 7: Plot Average Sales\nNow, we will plot the average sales as a line connecting the averages across the box plots.\n\n```python\n# Add average sales line\nfor index, row in average_sales.iterrows():\n    plt.plot([index, index], [0, row[\'Sales\']], color=\'red\', marker=\'o\', markersize=8, label=\'Average\' if index == 0 else "")\n```\n\n### Step 8: Customize the Plot\nAdd titles, labels, and a legend to make the chart more informative.\n\n```python\n# Customize the plot\nplt.title(\'Sales Distribution and Average Sales by Mobile Phone Brand\')\nplt.xlabel(\'Mobile Phone Brand\')\nplt.ylabel(\'Sales\')\nplt.xticks(rotation=45)\nplt.legend(title=\'Average Sales\', loc=\'upper right\')\nplt.grid(True)\n```\n\n### Step 9: Show the Plot\nFinally, display the plot.\n\n```python\n# Show the plot\nplt.tight_layout()\nplt.show()\n```\n\n### Complete Code\nHere is the complete code for your reference:\n\n```python\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Load the dataset\ndata = pd.read_csv(\'data.csv\')\n\n# Reshape the data to long format\ndata_long = data.melt(id_vars=\'Quarter\', var_name=\'Brand\', value_name=\'Sales\')\n\n# Calculate average sales for each brand\naverage_sales = data_long.groupby(\'Brand\')[\'Sales\'].mean().reset_index()\n\n# Set the color palette\npalette = sns.color_palette("husl", len(data[\'Quarter\'].unique()))\n\n# Create the box plots\nplt.figure(figsize=(12, 6))\nsns.boxplot(x=\'Brand\', y=\'Sales\', data=data_long, palette=palette)\n\n# Overlay individual sales data points\nsns.stripplot(x=\'Brand\', y=\'Sales\', data=data_long, color=\'black\', alpha=0.5, jitter=True)\n\n# Add average sales line\nfor index, row in average_sales.iterrows():\n    plt.plot([index, index], [0, row[\'Sales\']], color=\'red\', marker=\'o\', markersize=8, label=\'Average\' if index == 0 else "")\n\n# Customize the plot\nplt.title(\'Sales Distribution and Average Sales by Mobile Phone Brand\')\nplt.xlabel(\'Mobile Phone Brand\')\nplt.ylabel(\'Sales\')\nplt.xticks(rotation=45)\nplt.legend(title=\'Average Sales\', loc=\'upper right\')\nplt.grid(True)\n\n# Show the plot\nplt.tight_layout()\nplt.show()\n```\n\n### Conclusion\nThis code will generate a combination chart with box plots for each mobile phone brand, displaying individual sales data points and average sales lines. Adjust the color palette and other parameters as needed to fit your specific requirements.\n- plot_file_name:\nplot.png\n\nYour answer:\n', "Reasoning: Let's solve this problem step by step: \n"])
]

chat_prompt_template = ChatPromptTemplate.from_messages(msgs)

from langchain_openai.chat_models import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

routine = chat_prompt_template | model
# print(routine.invoke({}))
# print(model.invoke(msgs))

from langchain_core.messages import (
    merge_message_runs,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
)

messages = [
    SystemMessage("you're a good assistant."),
    SystemMessage("you're quite, only respond in 1 word."),
    HumanMessage(content="what's your favorite color", id="foo",),
    HumanMessage(content=
        [{"type": "text", "text": "wait your favorite food"}], id="bar",),
    AIMessage(
        "my favorite colo",
    ),
    AIMessage(
        [{"type": "text", "text": "my favorite dish is lasagna"}],
    ),
]

mmsg = merge_message_runs(messages)
print(mmsg)
print(model.invoke(messages).content)