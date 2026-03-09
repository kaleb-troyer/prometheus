# Exposes the `load` function, which returns a dataclass
# given data and the corresponding schema. Both the data
# and schema may be found under config/.
# 2026-03-09
# Kaleb Troyer

from dataclasses import is_dataclass, fields
from schema import *
from inputs import *

def load(cls, data: dict):
    """
    Recursively convert a nested dict into a dataclass
    instance using the provide schema.
    """
    if not is_dataclass(cls):
        return data

    kwargs = {}
    for f in fields(cls):
        key = f.name
        if key not in data:
            raise ValueError(f"Missing required key {key} for dataclass {cls.__name__}.")
        val = data[key]
        if val is not None and not isinstance(val, (f.type, dict)):
            if f.type is float and isinstance(val, int):
                pass
            else: raise TypeError(f"Value {f.name} type is {type(val)} but {f.type} was expected.")

        # recursively load all substructures
        kwargs[key] = load(f.type, val)

    return cls(**kwargs)

if __name__=='__main__':

    myclass = load(Receiver, RECEIVER)
    print(myclass)

# EOF
