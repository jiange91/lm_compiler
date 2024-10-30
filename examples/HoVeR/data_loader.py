import random
import json
import pandas as pd
from compiler.optimizer.registry import register_data_loader

    
train_size = 100
val_size = 50
dev_size = 200

@register_data_loader
def load_data():
    data_path = '/mnt/ssd4/lm_compiler/clean_examples/HoVeR/hover_data/hover/train/qas.json'
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            obj = json.loads(line)
            if obj['num_hops'] == 3:
                data.append(obj)
                
    rng = random.Random(0)
    rng.shuffle(data)
    
    train_set = data[:train_size]
    val_set = data[train_size:train_size+val_size]
    dev_set = data[train_size+val_size:train_size+val_size+dev_size]
    
    def input_ground_truth(x):
        return x['question'], x['support_pids']
    
    train_set = [input_ground_truth(x) for x in train_set]
    val_set = [input_ground_truth(x) for x in val_set]
    dev_set = [input_ground_truth(x) for x in dev_set]
    return train_set, val_set, dev_set