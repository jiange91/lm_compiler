class A:
    def __init__(self, data):
        self.data = data

class B:
    def __init__(self) -> None:
        self._a = A(10)
    
    @property
    def a(self):
        return self._a

b = B()
ob = B()

ob._a = b.a

b._a = A(20)

print(ob.a.data)
