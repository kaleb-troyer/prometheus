
from dataclasses import dataclass

# Maps the structure and contents of all input
# parameters. If a parameter is included in inputs.py
# but not in schema.py, it will not be delivered to
# the program.

#----------------------#
#---Receiver  Schema---#
#----------------------#

@dataclass
class ReceiverPanels:
    length: float
    height: float

@dataclass
class ReceiverLTE:
    desired: float
    trigger: float
    cutoff: float

@dataclass
class TubeOptions:
    is_adjacent_tubes: bool

@dataclass
class ReceiverTube:
    OD: float
    tw: float
    bend90: float
    bend45: float
    material: str

    # substructures
    options: TubeOptions

@dataclass # Primary Class
class Receiver:
    W_dot_net: float
    T_htf_i: float
    T_htf_o: float
    diameter: float
    length: float
    height: float
    use_sp_field: bool
    htower: float
    fluxmax: float
    flux_ub: float
    flux_lb: float
    htf_mat: str
    sol_abs: float
    emissivity: float
    heat_loss: float
    start_pt: str
    npaths: int
    ncross: int
    sig_lim_x: int
    sig_lim_y: int
    is_bottom_inlet: bool
    use_aiming_scheme: bool
    use_sp_flux: bool
    is_cross_too_high: bool
    is_min_before_cross: bool
    is_skip_panels: bool
    zenith: float
    azimuth: float
    offset_x: float
    offset_y: float
    offset_z: float
    aim_rows: int
    aim_cols: int

    # substructures
    panel: ReceiverPanels
    tube: ReceiverTube
    lte: ReceiverLTE

#----------------------#
#---Heliostat Schema---#
#----------------------#

@dataclass
class HeliostatImage:
    cutoff: float

@dataclass # Primary Class
class Heliostats:
    length: float
    height: float
    offset: float
    ncut: int
    model: str
    method: str
    err_refl_x: float
    err_refl_y: float
    err_surf_x: float
    err_surf_y: float

    # substructures
    img: HeliostatImage

#----------------------#
#---Plotting  Schema---#
#----------------------#

@dataclass # Primary Class
class Plot:
    fontsize: int
    dpi: int

#----------------------#
#---Optimizer Schema---#
#----------------------#

@dataclass # Primary Class
class Optimizer:
    f_og: float
    max_iter: int

# EOF
