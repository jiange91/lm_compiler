import copy
from concurrent.futures import ThreadPoolExecutor
from compiler.IR.base import Module, StatePool
import threading


def _forward(dm, **kwargs):
    result = dm.kernel(**kwargs)
    dm.hist.append(result)
    
    m_hist = dm.get_hist()
    dm.step_info.append(m_hist[-1])
    return result

_thread_local_chain = threading.local()

class Dummy(Module):
    def __init__(self, name, kernel):
        super().__init__(name, kernel)
        self.hist = []
        self.step_info = []
        setattr(_thread_local_chain, name, self)
    
    def get_hist(self):
        hist_cpy = copy.deepcopy(self.hist)
        self.hist = []
        return hist_cpy

    def get_thread_local_chain(self):
        try:
            if not hasattr(_thread_local_chain, self.name):
                setattr(_thread_local_chain, self.name, copy.deepcopy(self))
            return getattr(_thread_local_chain, self.name)
        except Exception as e:
            print(e)
            raise
    
    def forward(self, **kwargs):
        _self = self.get_thread_local_chain()
        return _forward(_self, **kwargs)

def foo(a):
    return {'b': a + 1}

dummy = Dummy('dummy', foo)

inputs = []
for i in range(100):
    statep = StatePool()
    statep.init({'a': i})
    inputs.append(statep)

def run(input: StatePool):
    dummy.invoke(input)
    try:
        print(input.news('b'))
    except Exception as e:
        print(e)
        raise

    
with ThreadPoolExecutor(10) as executor:
    for input in inputs:
        executor.submit(run, input)
# for input in inputs:
#     run(input)
print(dummy.step_info)       