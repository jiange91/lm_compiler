class MyClass:
    def __init__(self, attr):
        self.attr = attr

    def create_method(self):
        # Define the function code, referring to `self` directly
        func_str = """
def new_func():
    print(self.attr)
"""
        # Execute the function code within a context where `self` is available
        local_namespace = {}
        exec(func_str, {'self': self}, local_namespace)
        
        # Retrieve the function
        new_func = local_namespace['new_func']
        
        return new_func

# Example usage
obj = MyClass(attr="Hello, world!")

# Create the method
dynamic_method = obj.create_method()

# Call the dynamically created function
dynamic_method()  # Output: Hello, world!
