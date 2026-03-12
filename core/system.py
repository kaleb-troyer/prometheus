# <desc>
# 2026-03-11
# Kaleb Troyer

import pandas as pd
import numpy as np
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict, is_dataclass
from copy import deepcopy

import core.wenner.informed_aiming as ia
import core.wenner.thermal_model as tm
import core.wenner.damage_tool as dt
import config.inputs as ins
import config.schema as schema
import core.loader
from core.wenner.tube_jwenner import Tube

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
        self.sys = core.loader.load(schema.System, ins.SYSTEM)
        self.rec = core.loader.load(schema.Receiver, ins.RECEIVER)
        self.hel = core.loader.load(schema.Heliostats, ins.HELIOSTATS)
        self.plt = core.loader.load(schema.Plot, ins.PLOT)
        self.opt = core.loader.load(schema.Optimizer, ins.OPTIMIZER)

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
        self.sys = core.loader.load(schema.System, data['sys'])
        self.rec = core.loader.load(schema.Receiver, data['rec'])
        self.hel = core.loader.load(schema.Heliostats, data['hel'])
        self.plt = core.loader.load(schema.Plot, data['plt'])
        self.opt = core.loader.load(schema.Optimizer, data['opt'])

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

class Case():

    # tasks
    # - [x] instantiate the thermal model from user inputs
    # - [ ] generate a field from user inputs
    # - [ ] dynamically assign receiver as flat plate or cylindrical
    # - [ ] create and assign vars rec.type and rec.layout for billboard_receiver.py model
    # - [ ] generate a set of flux images from inputs
    # - [ ] identify case-driving inputs
    # - [ ] generate hash ONLY from schema vars
    # - [ ] light hash databasing
    # - [ ] review and incorporate Akshay's optimization

    # order of operations
    # - store hash/case information in db
    # - generate field layout if none exists
    # - generate flux images if none exists

    def __init__(self, par: Parameters):

        # these are shared references, not copies
        self.sys = par.sys
        self.rec = par.rec
        self.hel = par.hel
        self.hash = par.hash

        # The damage tool's primary purpose is to provide life-
        # time estimates using the products of the thermal tool,
        # namely fluid temperatures, temperature differences,
        # and the ratio of conductance to convective heat transfer.
        self.dmg = dt.damageTool(self.rec.tube.material)

        # The tube model provides material property functions
        # for the htf and wall alloy, acts as a data container
        # for tube geometry parameters, and provides methods
        # for calculating internal tube heat transfer coeffs.
        self.tube = Tube()
        # Design parameters are assigned to the tube, here.
        self.tube.OD = self.rec.tube.OD
        self.tube.twall = self.rec.tube.tw
        self.tube.roughness = self.rec.tube.roughness
        self.tube.solar_abs = self.rec.sol_abs
        self.tube.emis = self.rec.emissivity
        self.tube.HTF_material_name = self.rec.htf_mat
        self.tube.tube_material_name = self.rec.tube.material
        self.tube.tube_bends_45 = self.rec.tube.bend45
        self.tube.tube_bends_90 = self.rec.tube.bend90
        self.tube.initialize()

        # The thermal model solves for receiver tube and thermal
        # carrier temperatures (including heat transfer coeffs.),
        # given a flux profile and many design parameters.

        # !!! TEMPORARY VARIABLE BYPASS !!!
        self.rec.type = "temp none"
        self.rec.layout = "temp none"

        self.thermal_model = tm.setup_LWT_thermal_model_troyer(
            self.sys, self.rec, self.hel
        )

    def copy(self):
        return deepcopy(self)

    def get_lifetime_estimate(self):

        dTs, Tfs, qabs, Rs = tm.solve_LWT_thermal_model(
            self.thermal_model, self.fluxgrid
        )

        LTEs = self.dmg.get_LTEs(
            dTs.flatten(), Tfs.flatten(), Rs.flatten()
        )

        min_panel_LTEs, min_tube_LTEs = dmg.calc_minimum_panel_LTEs(
            self.thermal_model, LTEs
        )

        # and then do something with them

    def _generate_field_layout(self):

        # example in jacob's code?
        pass

    def _generate_flux_images(self):

        # example in jacob's code?
        pass

    def _generate_ideal_flux_grid(self):

        # !!!! also need to set this test up
        if file_doesnt_exist:

            ideal_fluxmap, _, _ = ia.generate_ideal_fluxmap_with_offset(
                self.dmg, self.tube,
                self.rec.lte.desired,
                self.rec.mflow,
                self.rec.panel.length,
                self.rec.panel.height,
                self.rec.aim_marg,
                self.rec.start_pt
            )

            ia.save_ideal_fluxmap(
                ideal_fluxmap, os.path.join(
                    os.getcwd(), 'case', self.hash, 'ideal_flux_map.json'
                )
            )

        self.ideal_fluxgrid = ia.build_ideal_fluxgrid(
            os.path.join(os.getcwd(), 'case', self.hash, 'ideal_flux_map.json'),
            res_y = self.thermal_model.Npanels,
            H = self.thermal_model.tubes[0][0].length,
            W = self.thermal_model.D / self.thermal_model.Npanels,
            flowpath_config = self.rec.start_pt
        )

        # increase fluxgrid resolution for SolarPILOT / damage tool?

        self.fluxgrid = self.ideal_fluxgrid


if __name__=='__main__':

    par = Parameters().load()
    des = Case(par)

    print('\nCase and parameters successfully instantiated!\n')

# EOF
