import functools
import copy

_reg_opt_modules_ = {}
_reg_opt_program_entry_ = None
_reg_opt_score_fn_ = None

def clear_registry():
    global _reg_opt_program_entry_
    global _reg_opt_score_fn_
    _reg_opt_modules_.clear()
    _reg_opt_program_entry_ = None
    _reg_opt_score_fn_ = None

def register_opt_module(module):
    _reg_opt_modules_[module.name] = module

def register_opt_program_entry(program):
    global _reg_opt_program_entry_
    _reg_opt_program_entry_ = program
    return program

def register_opt_score_fn(score_fn):
    global _reg_opt_score_fn_
    _reg_opt_score_fn_ = score_fn
    return score_fn
    
def get_registered_opt_modules():
    return list(_reg_opt_modules_.values())

def get_registered_opt_program_entry():
    return _reg_opt_program_entry_

def get_registered_opt_score_fn():
    return _reg_opt_score_fn_