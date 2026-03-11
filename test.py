
from copy import deepcopy
import pandas as pd
import numpy as np
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict, is_dataclass
from config.inputs import *
from config.schema import *
from core.loader import *

class Parameters():

    def __init__(self):

        # Design parameters are loaded using `inputs.py` and `schema.py`.
        # The schema insures only expected parameters of the appropriate
        # data type are passed along to the rest of the program.
        self.sys = {}
        self.rec = {}
        self.hel = {}
        self.plt = {}
        self.opt = {}

        # This hash is a unique str representation of the case parameters which
        # determine solar field layout, heliostat flux images, and the ideal
        # fluxmap. Most (but not all) design parameters are used in the hash.
        # The parameters not included are all optimization parameters and
        # aimpoint strategy parameters. The hash is created using the `freeze()`
        # method, which also creates the case directory and saves the parameters
        # as a json file for record keeping.
        self.hash = None

    def load(self):
        self.sys = load(System, SYSTEM)
        self.rec = load(Receiver, RECEIVER)
        self.hel = load(Heliostats, HELIOSTATS)
        self.plt = load(Plot, PLOT)
        self.opt = load(Optimizer, OPTIMIZER)

        self.freeze()
        return self

    def copy(self):
        pars = deepcopy(self)
        pars.hash = None
        return pars

    def from_json(self, hash):
        path = os.path.join(os.getcwd(), 'cases', hash)
        file = 'inputs.json'
        with open(os.path.join(path, file), 'r') as file:
            data = json.load(file)
        self.sys = load(System, data['sys'])
        self.rec = load(Receiver, data['rec'])
        self.hel = load(Heliostats, data['hel'])
        self.plt = load(Plot, data['plt'])
        self.opt = load(Optimizer, data['opt'])
        self.freeze()
        return self

    def freeze(self):
        self.hash = self._generate_case_hash(
            self.sys.solar_resource, self.sys.hour_idx, self.sys.des_dni,
            self.rec, self.hel
        )
        self._mkdir()
        self._to_json()

    def _generate_case_hash(self, *args):
        parameters = [asdict(obj) if is_dataclass(obj) else obj for obj in args]
        serialized = json.dumps(parameters, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()[:16]

    def _mkdir(self):
        if self.hash == None:
            raise ValueError("Must generate hash using the freeze() method before creating the directory.")
        os.makedirs(
            os.path.join(os.getcwd(), 'cases', self.hash),
            exist_ok=True
        )

    def _to_json(self):
        if self.hash == None:
            raise ValueError("Must generate hash using the freeze() method before saving the parameters as a json file.")
        pars = {
            'sys': asdict(self.sys),
            'rec': asdict(self.rec),
            'hel': asdict(self.hel),
            'plt': asdict(self.plt),
            'opt': asdict(self.opt)
        }
        with open(os.path.join(os.getcwd(), 'cases', self.hash, 'inputs.json'), 'w') as file:
            json.dump(pars, file, indent=4)

if __name__=='__main__':

    par = Parameters().load()
    foo = Parameters().from_json(par.hash)
    print(par.hash)
    print(foo.hash)

# EOF
