from compiler.optimizer.registry import register_data_loader
from dspy.datasets.hotpotqa import HotPotQA

@register_data_loader
def load_data():
    dataset = HotPotQA(train_seed=1, train_size=150, eval_seed=2023, dev_size=200, test_size=0)
    def get_input_label(x):
        return x.question, x.answer
    trainset = [get_input_label(x) for x in dataset.train[0:10]]
    valset = [get_input_label(x) for x in dataset.train[100:115]]
    devset = [get_input_label(x) for x in dataset.dev[:10]]
    return trainset, valset, devset
