from dataclasses import dataclass
import copy

@dataclass
class A:
    a: int
    
a = A(10)

@dataclass
class C:
    a: A
    

class B:
    def __init__(self):
        self.a = a
        self.c = C(a)
        
b = B()

b_copy = copy.deepcopy(b)
b_copy.a.a = 20

print(a)
print(b)
print(b_copy)
print(b.c)
print(b_copy.c)