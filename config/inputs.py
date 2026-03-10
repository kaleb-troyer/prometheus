
# This file is the complete set of aggregated inputs
# for PROMETHEUS. All data is stored in python dictionaries,
# validated by schema.py, and parsed by loader.py.
#
#  NOTE: Many variables are inherited from Janna's original thermal
#        model and are defined under "receiver.py" and "tube.py"

UNCATEGORIZED = {
    "vel_ref": 3.298, # [m/s] deprecated artifact variable for automatically setting number of receiver panels to meet a design velocity
    "n_circulation": 0, # [?] ???
    "m_comb": 3.2, # [?] possibly a variable of Janna's receiver thermal model

    # unused / repeat variables
    "flow_paths": 2,  # [-] duplicate variable of Janna's npaths
    "aim_marg_h": 0,  # [?] unused by Jacob, possibly a HALOS parameter for minimum edge offset
    "aim_marg_v": 0,  # [?] unused by Jacob, possibly a HALOS parameter for minimum edge offset
    "dresize": False, # [bool] triggers a resizing of the receiver in Janna's model if flux is too high
    "sec_num":    7,       # [-] HALOS parameter IF you want it to optimize aimpoints (Jacob's code avoids this)
    "sec_method": "angle", # [-] HALOS parameter for splitting field into subfields to accelerate optimization
}

SYSTEM = {
    "min_turndown": 0.25,

    "pump_efficiency": 0.85,
    "cycle_efficiency": 0.412,
    "solar_resource": "USA CA Daggett (TMY2).csv",
    "hour_idx": 4090, # [hour] hour of the meteorogical year to consider weather and solar position
    "des_dni": 950,
}

RECEIVER = {
    # Power and temperature requirements
    "W_dot_net": 175,  # [MWt] net power absorbed by the receiver
    "T_htf_i":   563,  # [C] heat transfer fluid inlet temperature
    "T_htf_o":   838,  # [C] heat transfer fluid outlet temperature
    "fluxmax":   1000, # [kW/m2] target maximum flux on the receiver
    "flux_ub":   600,  # [kW/m2] upper limit on incident receiver flux
    "flux_lb":   0,    # [kW/m2] lower limit on incident receiver flux

    # Receiver operation, fluid, and performance
    "htf_mat":    "Salt_60NaNO3_40KNO3", # [-] thermal carrier (htf) material
    "sol_abs":    0.96, # [-] receiver solar absorptivity
    "emissivity": 0.87, # [-] receiver total emissivity
    "heat_loss":  30,   # [kW/m2] estimated receiver heat loss by area
    "start_pt":   "ctr",# [-] receiver flow path starting point (e.g. ctr is for center, top of flat plate receiver)
    "npaths":     2,    # [-] number of parallel flow paths in the receiver (typically 2)
    "ncross":     0,    # [-] count of times a flow path crosses from one side of the receiver to the other
    "sig_lim_x":  2,    # [-] minimum flux image standard deviations the image may be from vertical edges of the receiver during placement
    "sig_lim_y":  2,    # [-] minimum flux image standard deviations the image may be from horizontal edges of the receiver during placement

    # Receiver discretized aimpoint parameters
    "aim_rows":   7,    # [-] specifies number of vertical aimpoint options for HALOS to place images
    "aim_cols":   7,    # [-] specifies number of horizontal aimpoint options for HALOS to place images

    # Overall receiver geometry
    "diameter": None, # [m] receiver diameter, set to None if not cylindrical
    "length":   18,   # [m] total horizontal length of the receiver, set to None if not a flat plate
    "height":   15,   # [m] total vertical height of the receiver
    "htower":   170,  # [m] tower height
    "zenith":   90,   # [deg] specified position of the sun for solarpilot simulation
    "azimuth":  180,  # [deg] specified position of the sun for solarpilot simulation
    "offset_x": 0,    # [m] receiver x-axis offset
    "offset_y": 0,    # [m] receiver y-axis offset
    "offset_z": 0,    # [m] receiver z-axis offset

    # Receiver configuration options
    "use_sp_field":      False, # [bool] use SolarPILOT-generated field instead of internal layout
    "use_sp_flux":       True,  # [bool] use SolarPILOT-generated flux profile instead of aimpoint flux profile
    "is_bottom_inlet":   False, # [bool] does the htf enter from the bottom of the receiver?
    "use_aiming_scheme": False, # [bool] directs the thermal model to calculate flux profile through copylot simulation. if false, profile is uniform
    "is_cross_too_high": False, # [bool] cross receiver flow paths from low to high flux sides to balance energy transfer in thermal model
    "is_min_before_cross": False, # [?] ???
    "is_skip_panels":    False, # [?] ???

    # Panel-level geometry
    "panel": {
        "number": 18,   # [-] total count of panels that make up the receiver
        "length": None, # [m] horizontal length of a single panel
        "height": None, # [m] vertical height of a single panel
    },

    # Tube geometry and options
    "tube": {
        "OD": 0.05080,  # [m] outer diameter
        "tw": 0.00125,  # [m] wall thickness
        "bend45": 0,    # [-] number of 45 degree bends
        "bend90": 4,    # [-] number of 90 degree bends
        "material": "A230", # [-] tube wall material

        "options": {
            # Prevents initialization error when tubes are not adjacent
            "is_adjacent_tubes": False,
        },
    },

    # Lifetime estimate (LTE) parameters
    "lte": {
        "desired": 80,  # [yrs] target lifetime
        "trigger": 240, # [yrs] threshold to begin LTE-driven optimization
        "cutoff":  15,  # [yrs] hard cutoff for unacceptable designs
    },
}

HELIOSTATS = {
    # Sizing and selection parameters
    "length": 12.2, # [m] heliostat length on x-axis
    "height": 12.2, # [m] heliostat lenght on y-axis
    "offset": 12.2, # [m] heliostat offset from ground
    "ncut":   0,    # [-] number of rows in heliostat array that are discarded

    # Flux modeling parameters
    "model":      "SinglePointGaussian",  # [-] solarpilot flux modeling strategy (see 'help' page in SolarPILOT)
    "method":     "SimpleNormalFluxCalc", # [-] solarpilot flux calculation strategy (see 'help' page in SolarPILOT)

    # Reflectivity and surface error parameters
    "err_refl_x": 2e-4, # [?] heliostat reflectivity error in x (see solarpilot documentation)
    "err_refl_y": 2e-4, # [?] heliostat reflectivity error in y
    "err_surf_x": 2e-4, # [?] heliostat surface error in x (see solarpilot documentation)
    "err_surf_y": 2e-4, # [?] heliostat surface error in y

    # Flux image modifying parameters
    "img": {
        "cutoff": 0.01, # [-] minimum normalized flux to retain of discretized image
    },
}

PLOT = {
    "fontsize": 12, # [-] Axis label and numbering font size
    "dpi": 300,     # [-] Image dots-per-inch resolution
}

OPTIMIZER = {
    "f_og": 1,      # [?] ???
    "max_iter": 50, # [-] Maximum optimizer iterations
}

# EOF
