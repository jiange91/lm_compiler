import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__))))
from compiler.optimizer.registry import register_data_loader
from humaneval.humaneval import HumanEvalDataset
import random

@register_data_loader
def load_data():
    raw_dataset = HumanEvalDataset()
    size = len(raw_dataset.data)
    # shuffle the data
    random.seed(42)
    random.shuffle(raw_dataset.data)
    data = []
    for i in range(size):
        problem = raw_dataset.data[i]
        data.append((problem['prompt'], problem))
    train, val, test = data[:40], data[40:60], data[60:]
    return train, val, test
    
if __name__ == '__main__':
    train, val, dev = load_data()