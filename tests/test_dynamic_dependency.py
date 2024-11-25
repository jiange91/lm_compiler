from cognify.graph.program import Workflow, hint_possible_destinations
from cognify.graph.modules import CodeBox, StatePool, InputModule, Output
import unittest

class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, aggregate):
        print(f"Adding {self._value} to {aggregate}")
        aggregate.append(self._value)
        return {"aggregate": aggregate}

workflow = Workflow('test_dd')
workflow.add_module(CodeBox('a', ReturnNodeValue('Im A')))
workflow.add_module(CodeBox('b', ReturnNodeValue('Im B')))
workflow.add_module(CodeBox('b2', ReturnNodeValue('Im B2')))
workflow.add_module(CodeBox('c', ReturnNodeValue('Im C')))
workflow.add_module(CodeBox('d', ReturnNodeValue('Im D')))
workflow.add_module(CodeBox('e', ReturnNodeValue('Im E')))
workflow.add_module(InputModule('input'))
workflow.add_module(Output('output'))

@hint_possible_destinations(['b', 'c', 'd'])
def route_bc_or_cd(ctx, which) -> list[str]:
    if which == "cd":
        return ["c", "d"]
    return ["b", "c"]

workflow.add_edge('input', 'a')
workflow.add_branch('bc_or_cd', 'a', route_bc_or_cd)
workflow.add_edge('b', 'b2')
workflow.add_edge(['b2', 'c', 'd'], 'e')
workflow.add_edge('e', 'output')

workflow.compile()


class TestDynamicDependency(unittest.TestCase):
    def test_dynamic_dependency_check(self):
        statep = StatePool()
        statep.init({'aggregate': [], 'which': 'bc'})

        workflow.pregel_run(statep)
        self.assertCountEqual(statep.news('aggregate'), ['Im A', 'Im C', 'Im B', 'Im B2', 'Im E'])
        self.assertEqual(workflow.current_step, 7)
        workflow.reset()