from compiler.optimizer.registry import register_data_loader
from dspy.datasets.hotpotqa import HotPotQA

@register_data_loader
def load_data_minor():
    trainset = [
        ("""Are Walt Disney and Sacro GRA both documentry films?""", """yes"""),
        ("""What do students do at the school of New York University where Meleko Mokgosi is an artist and assistant professor?""", """design their own interdisciplinary program"""),
        ("""Which is published more frequently, The People's Friend or Bust?""", """The People's Friend"""),
        ("""How much is spent on the type of whiskey that 1792 Whiskey is in the United States?""", """about $2.7 billion"""),
        ("""The place where John Laub is an American criminologist and Distinguished University Professor in the Department of Criminology and Criminal Justice at was founded in what year?""", """1856"""),
        ("""What year did the mountain known in Italian as "Monte Vesuvio", erupt?""", """79 AD"""),
        ("""What was the full name of the author that memorialized Susan Bertie through her single volume of poems?""", """Emilia Lanier"""),
        ("""How many seasons did, the Guard with a FG%% around .420, play in the NBA ?""", """14 seasons"""),
        ("""Estonian Philharmonic Chamber Choir won the grammy Award for Best Choral Performance for two songs by a composer born in what year ?""", """1935"""),
        ("""Which of the sport analyst of The Experts Network is nicknamed  "The Iron Man"?""", """Calvin Edwin Ripken Jr."""),
        ("""What are both National Bird and America's Heart and Soul?""", """What are both National Bird and America's Heart and Soul?"""),
        ("""What was the 2010 population of the birthplace of Gerard Piel?""", """17,121"""),
        ("""On what streets is the hospital that cared for Molly Meldrum located?""", """the corner of Commercial and Punt Roads"""),
    ]
    return trainset[:3], trainset[3:5], trainset[0:1]

def load_data():
    dataset = HotPotQA(train_seed=1, train_size=150, eval_seed=2023, dev_size=200, test_size=0)
    def get_input_label(x):
        return x.question, x.answer
    trainset = [get_input_label(x) for x in dataset.train[0:100]]
    valset = [get_input_label(x) for x in dataset.train[100:150]]
    devset = [get_input_label(x) for x in dataset.dev]
    print(len(trainset), len(valset), len(devset))
    return trainset, valset, devset