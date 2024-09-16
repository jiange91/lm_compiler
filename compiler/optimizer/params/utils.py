from .common import ParamBase, OptionBase
import json

def dump_params(params: list[ParamBase], log_path: str):
    print(f'---- Dumping parameters to {log_path} ----')
    ps = [param.to_dict() for param in params]
    with open(log_path, 'w+') as f:
        json.dump(ps, f, indent=4)
        
def load_params(log_path: str) -> list[ParamBase]:
    print(f'---- Loading parameters from {log_path} ----')
    with open(log_path, 'r') as f:
        data = json.load(f)
    params = []
    for dat in data:
        t = ParamBase.registry[dat['type']]
        params.append(t.from_dict(dat))
    return params