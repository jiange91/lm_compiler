import toml
import sys
import os
import functools
import warnings
import logging
    
        
def load_api_key(toml_file_path):
    try:
        with open(toml_file_path, 'r') as file:
            data = toml.load(file)
    except FileNotFoundError:
        print(f"File not found: {toml_file_path}", file=sys.stderr)
        return
    except toml.TomlDecodeError:
        print(f"Error decoding TOML file: {toml_file_path}", file=sys.stderr)
        return
    # Set environment variables
    for key, value in data.items():
        os.environ[key] = str(value)
            
def deprecate_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        raise DeprecationWarning(f"{func.__name__} is deprecated")
    return wrapper
