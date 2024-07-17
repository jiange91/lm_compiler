from collections import defaultdict
from compiler.IR.program import StatePool

# Define State
class StormState(StatePool):
    def __init__(self):
        super().__init__()
    
    def dump(self, path: str):
        pass
    
    def load(self, path: str):
        pass