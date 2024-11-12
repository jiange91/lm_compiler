from cognify.IR.program import Workflow, hint_possible_destinations, Context
from cognify.IR.modules import CodeBox, StatePool, Input, Output
import unittest

class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, aggregate):
        print(f"Adding {self._value} to {aggregate}")
        aggregate.append(self._value)
        return {"aggregate": aggregate}

workflow = Workflow()
workflow.add_module(CodeBox('a', ReturnNodeValue('Im A')))
workflow.add_module(CodeBox('b', ReturnNodeValue('Im B')))
workflow.add_module(CodeBox('c', ReturnNodeValue('Im C')))
workflow.add_module(CodeBox('d', ReturnNodeValue('Im D')))
workflow.add_module(Input('input'))
workflow.add_module(Output('output'))

@hint_possible_destinations(['output', 'b'])
def accept_or_not(retry) -> list[str]:
    ctx: Context = accept_or_not.ctx
    if ctx.invoke_time >= retry:
        return 'output'
    return 'b'

workflow.add_edge('input', 'a')
workflow.add_edge('a', 'b')
workflow.add_edge('b', 'c')
workflow.add_edge('c', 'd')
workflow.add_branch('d', accept_or_not)

workflow.compile()

# statep = StatePool()
# statep.init({'aggregate': [], 'retry': 2})

# workflow.pregel_run(statep)
# print(statep.all_news())

class TestCycles(unittest.TestCase):
    def test_cycles(self):
        statep = StatePool()
        statep.init({'aggregate': [], 'retry': 2})

        workflow.pregel_run(statep)
        self.assertCountEqual(statep.news('aggregate'), ['Im A', 'Im B', 'Im C', 'Im D', 'Im B', 'Im C', 'Im D'])
        self.assertEqual(workflow.current_step, 9)
        workflow.reset()