# <desc>
# 2026-03-11
# Kaleb Troyer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import hashlib
import json
import math
import os
from dataclasses import dataclass, field, asdict, is_dataclass
from copy import deepcopy

import core.wenner.informed_aiming as ia
import core.wenner.thermal_model as tm
import core.wenner.damage_tool as dt
import config.schema as schema
import config.inputs as ins
import core.loader
from core.wenner.tube_jwenner import Tube

class Parameters():

    def __init__(self):

        # Design parameters are loaded using `inputs.py` and `schema.py`.
        # The schema ensures only expected parameters of the appropriate
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

        self._processing()
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

        self._processing()
        self.freeze()
        return self

    def freeze(self):
        # Each parameter must be explicitly declared here, as opposed to passing the
        # entire dataclass, because parameters are naturally extended and copied in
        # the program. Otherwise, extending the class and rehashing would produce a 
        # different hash where it should not.
        self.hash = self._generate_case_hash(
            self.sys.min_turndown,      # not sure if actually used in thermal model
            self.sys.pump_efficiency,   # not sure if actually used in thermal model
            self.sys.cycle_efficiency,  # not sure if actually used in thermal model
            self.sys.solar_resource, self.sys.hour_idx, self.sys.des_dni,
            self.sys.t_amb, self.sys.rel_hum, self.sys.vwind10,

            self.rec.W_dot_net, self.rec.T_htf_i, self.rec.T_htf_o,
            self.rec.fluxmax, self.rec.flux_ub, self.rec.flux_lb,
            self.rec.htf_mat, self.rec.sol_abs, self.rec.emissivity,
            self.rec.heat_loss, self.rec.m_comb, self.rec.start_pt,
            self.rec.npaths, self.rec.ncross, self.rec.ntubesim,
            self.rec.aim_marg, self.rec.diameter, self.rec.length,
            self.rec.height, self.rec.htower, self.rec.zenith,
            self.rec.azimuth, self.rec.offset_x, self.rec.offset_y,
            self.rec.offset_z, self.rec.panel.number, self.rec.tube.OD,
            self.rec.tube.tw, self.rec.tube.bend45, self.rec.tube.bend90,
            self.rec.tube.material, self.rec.tube.roughness,
            self.rec.tube.options.is_adjacent_tubes, self.rec.lte.desired,

            self.hel.length, self.hel.height, self.hel.offset,
            self.hel.ncut, self.hel.model, self.hel.method, 
            self.hel.err_refl_x, self.hel.err_refl_y,
            self.hel.err_surf_x, self.hel.err_surf_y,
            self.hel.img.cutoff
        )

        self._mkdir()
        self._to_json()

    def _processing(self):
        if self.rec.diameter != None and self.rec.length != None:
             ValueError("Cannot have both diameter and length.")
        elif self.rec.diameter == None and self.rec.length == None:
            raise ValueError("Must have a value for diameter or length.")
        elif self.rec.diameter != None:
            self.rec.length = math.pi * self.rec.diameter
            self.rec.type = "cylinder"
            self.rec._len_derived = True
        else: self.rec.type = "flat plate"

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
    # - [x] dynamically assign receiver as flat plate or cylindrical
    # - [x] create and assign vars rec.type and rec.layout for billboard_receiver.py model
    # - [x] identify case-driving inputs
    # - [x] generate hash ONLY from schema vars
    # - [x] generate and save the ideal fluxmap and grid
    # - [ ] generate a field from user inputs
    # - [ ] generate a set of flux images from inputs
    # - [ ] don't let thermal model assume typical temperature range
    # - [ ] vet case driving inputs (what variables actually produce different results?)
    # - [ ] review and incorporate Akshay's optimization
    # - [ ] light hash databasing

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
        self.thermal_model = tm.setup_LWT_thermal_model_troyer(
            self.sys, self.rec, self.hel
        )

    def copy(self):
        return deepcopy(self)

    def get_lifetime_estimate(self, fluxgrid=None):

        if fluxgrid == None:
            fluxgrid = self.fluxgrid

        dTs, Tfs, qabs, Rs = tm.solve_LWT_thermal_model(
            self.thermal_model, fluxgrid
        )

        LTEs = self.dmg.get_LTEs(
            dTs.flatten(), Tfs.flatten(), Rs.flatten()
        )

        min_panel_LTEs, min_tube_LTEs = dmg.calc_minimum_panel_LTEs(
            self.thermal_model, LTEs
        )

        # and then do something with them. for now:
        return min_panel_LTEs

    def _generate_field_layout(self):

        # example in jacob's code?
        pass

    def _generate_flux_images(self):

        # example in jacob's code?
        pass

    def _generate_ideal_flux_grid(self):

        # inherited problems with this function:
        # - Jacob's code is not designed to use anything other than
        #   two flowapths with a center starting point. The options are
        #   presented as design parameters but meaningless.
        # - rec.panel.number, while used to derive rec.panel.length,
        #   has apparently no bearing on the design. It does not
        #   impact the ideal number of panels despite affecting
        #   the size of the panel, and the difference between the
        #   two is not clear.
        # - An informed aimer function `generate_ideal_fluxmap_with_offset`
        #   proports (in Jacob's thesis) to improve performance by
        #   enforcing an offset region where heliostats are not allowed
        #   to aim. I haven't gotten this function to work in practice.
        #   To convert the fluxmap to a fluxgrid, it must be saved as
        #   a json to use the `build_ideal_fluxgrid` fucntion. However, the
        #   function in question stores some objects as lambda functions,
        #   which cannot be saved in the json format. I haven't tried
        #   modifying the `build_ideal_fluxgrid` to take the fluxmap directly.

        path = os.path.join(os.getcwd(), 'cases', self.hash)
        file = 'ideal_flux_map.json'
        fluxmap_exists = os.path.exists(
            os.path.join(path, file)
        )

        self.rec.mflow = np.sum(self.thermal_model.operating_conditions.mass_flow)
        self.rec.panel.length = self.rec.length / self.rec.panel.number
        self.rec.panel.height = self.rec.height
        self.rec.fp_config = f"{self.rec.npaths}_{self.rec.start_pt}"

        if not fluxmap_exists:

            # This function produces the ideal fluxmap. A fluxmap is
            # distinct from a fluxgrid in that a fluxgrid has uniformly
            # distributed points, adapted for use by SolarPILOT.
            ideal_fluxmap, _, _ = ia.generate_ideal_fluxmap(
                self.dmg, self.tube,
                self.rec.lte.desired,
                self.rec.mflow,         # derived
                self.rec.panel.length,  # derived
                self.rec.panel.height,  # derived
                self.rec.fp_config,     # derived
            )

            ia.save_ideal_fluxmap(
                ideal_fluxmap, os.path.join(path, file)
            )

        else: # need the fluxmap to extract total ideal panel count
            print("just loading the fluxmap")
            with open(os.path.join(path, file), 'r') as f:
                ideal_fluxmap = json.load(f)

        # I believe the distinction between rec.panels.number and
        # N_panels_ideal is that rec.panels.number is a budget, but
        # N_panels_ideal is all that is actually required. However,
        # you would expect the former to effect the latter, but I
        # have not observed this.
        if self.rec.fp_config == '2_ctr':
            N_panels_half  = len(ideal_fluxmap.keys())
            N_panels_ideal = N_panels_half*2

        self.ideal_fluxgrid = ia.build_ideal_fluxgrid(
            os.path.join(path, file),
            res_y = self.rec.panel.number,
            H = self.rec.panel.height,
            W = self.rec.panel.length * N_panels_ideal,
            flowpath_config = self.rec.fp_config
        )

        ################################################################
        ####                                                        ####
        ####    Code hereafter is a WIP  and not well understood    ####
        ####                                                        ####
        ################################################################

        # Jacob has this in his example. I'm not sure why the error occurs.
        # It makes the fluxgrid square by including unutilized panels.
        if N_panels_ideal != np.round(self.rec.length / self.rec.panel.length):
            print('Error! The minimum panel number for receiver @ this height differs from that in receiver object. adapting to match thermal model')
            self.ideal_fluxgrid = ia.fit_grid_to_receiver(
                self.ideal_fluxgrid, self.thermal_model, self.rec.fp_config
            )

        # this can be removed, its just here for testing
        ia.plot_ideal_fluxgrid(
            self.ideal_fluxgrid, self.rec.height, self.rec.length, self.rec.fp_config,
            savefig=False,
            display=True
        )

        ## testing only ################################################
        self.ideal_fluxgrid = tm.increase_flux_resolution_blocked(
            flux_low_res = self.ideal_fluxgrid, ndim_new = self.rec.panel.number * 3
        )

        self.fluxgrid = self.ideal_fluxgrid

        print()
        print()

        # the thermal model is the thing thats failing
        tm.solve_LWT_thermal_model(self.thermal_model, self.ideal_fluxgrid)
        dTs, Tfs, qabs, Rs = tm.get_thermal_results(self.thermal_model)

        # all of these should return as false
        print(np.any(np.isnan(dTs)), np.any(np.isinf(dTs)))
        print(np.any(np.isnan(Tfs)), np.any(np.isinf(Tfs)))
        print(dTs)
        quit()

        # revealed by this plot, temperature fluids from thermal model don't make any sense
        fig, ax = plt.subplots()
        im = ax.imshow(Tfs[:, :, 0], aspect='auto', origin='lower', cmap='hot')
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        ax.set_xlabel('Tube index')
        ax.set_ylabel('Axial node')
        ax.set_title('Fluid Temperature Distribution')
        plt.tight_layout()
        plt.show()

        quit()
        ################################################################


if __name__=='__main__':

    par = Parameters().load()
    des = Case(par)

    des._generate_ideal_flux_grid()
    LTEs = des.get_lifetime_estimate()
    print(LTEs)

    print('\nCase and parameters successfully instantiated!\n')

# EOF
