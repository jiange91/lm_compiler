from compiler.IR.program import Workflow, hint_possible_destinations
from compiler.IR.modules import CodeBox, StatePool, Input, Output
import unittest

class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, aggregate):
        print(f"Adding {self._value} to {aggregate}")
        aggregate.append(self._value)
        return {"aggregate": aggregate}

workflow = Workflow('test_md')
workflow.add_module(CodeBox('a', ReturnNodeValue('Im A')))
workflow.add_module(CodeBox('b', ReturnNodeValue('Im B')))
workflow.add_module(CodeBox('c', ReturnNodeValue('Im C')))
workflow.add_module(CodeBox('d', ReturnNodeValue('Im D')))
workflow.add_module(Input('input'))
workflow.add_module(Output('output'))

@hint_possible_destinations(['b', 'c'])
def route_b_or_c(ctx, which) -> list[str]:
    return which

workflow.add_edge('input', 'a')
workflow.add_branch('b_or_c', 'a', route_b_or_c)
workflow.add_edge('a', 'c', enhance_existing=True)
workflow.add_edge('c', 'output')
workflow.add_edge('b', 'output')

workflow.compile()
statep = StatePool()
statep.init({'aggregate': [], 'which': 'c'})

workflow.pregel_run(statep)


class TestMultipleDependency(unittest.TestCase):
    def test_dynamic_dependency_check(self):
        statep = StatePool()
        statep.init({'aggregate': [], 'which': 'c'})

        workflow.pregel_run(statep)
        self.assertCountEqual(statep.news('aggregate'), ['Im A', 'Im C', 'Im B', 'Im B2', 'Im E'])
        self.assertEqual(workflow.current_step, 7)
        workflow.reset()