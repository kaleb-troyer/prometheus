
from dataclasses import dataclass, field
from copy import deepcopy
import os
import pandas as pd
import numpy as np
from numpy.typing import NDArray

casename = 'compare_3D_heuristic_140'

class Prometheus():

    def __init__(self):

        self._declare_data_structures()

    def copy(self):
        return deepcopy(self)

    def _declare_data_structures(self):

        @dataclass
        class Paths():
            chapter3: str = os.path.join(os.getcwd(), 'wenner', 'Chapter_3')
            chapter4: str = os.path.join(os.getcwd(), 'wenner', 'Chapter_4')
            receiver: str = os.path.join(chapter4, 'receivers')
            casepath: str = os.path.join(chapter4, 'case_inputs')
            aiming:   str = os.path.join(chapter4, 'aiming')
            report:   str = os.path.join(chapter4, 'reports')
        self.paths = Paths()

        @dataclass
        class Files():
            inputs:   str = f'heuristic_receiver.json'
            output:   str = f'{casename}_results.json'
            casefile: str = f'{casename}.csv'
            fluxmap:  str = f'{casename}_3D_heuristic_dev_fluxmap.json'
            fluxgrid: str = f'ideal_fluxgrid_model_lite.Qdes_LTELTE_desired_for_SPT.csv'
            img_flux_prof: str = f'{casename}_flux_profile.png'
            img_placement: str = f'{casename}_image_placements.json'
        self.files = Files()

        @dataclass
        class Receiver():
            width:    float = 0.0
            height:   float = 0.0
            area:     float = 0.0
            massflow: float = 0.0
            material: str = ''

            @dataclass
            class Tube():
                OD: float = 0.05080
                tw: float = 0.00125

                @dataclass
                class Options():
                    is_adjacent_tubes: bool = False
                options: Options = field(default_factory=Options)
            tube: Tube = field(default_factory=Tube)

            @dataclass
            class LTE():
                desired:   float = 80
                trigger:   float = 240
                threshold: float = 15
            lte: LTE = field(default_factory=LTE)
        self.rec = Receiver()

        @dataclass
        class Heliostats():
            width:      float = None
            height:     float = None
            ncut:       int = 0
            aiming:     list = field(default_factory=list) # collection of heliostat receiver aimpoints
            placements: list = field(default_factory=list) # collection of heliostat field coordinates
            fluxmaps:   dict = field(default_factory=dict) # collection of heliostat flux-on-receiver maps, struct: {"<#id>" [[...], ...] (54x54)}
            fluxmodel:  Optional[object] = None

            @dataclass
            class Image(): # UNSURE HOW THIS IS TO BE USED YET
                threshold: float = 0.01
                widths:    dict = field(default_factory=dict)
                heights:   dict = field(default_factory=dict)
                rel_areas: NDArray = field(default_factory=lambda: np.array([]))
                keys:      list = field(default_factory=list)
            img: Image = field(default_factory=Image)
        self.hel = Heliostats()

        @dataclass
        class Plot():
            fontsize: int = 12
            dpi:      int = 300
        self.plt = Plot()

        @dataclass
        class Optimizer():
            t_i:           float = 0
            f_og:          float = 1
            m_iter:        int = 50
            T_out_obj:     float = 565
            is_lhs_done:   bool = True
            is_rhs_done:   bool = False
            in_loop:       bool = True
            is_f_violated: bool = False
            is_endgame:    bool = False
        self.opt = Optimizer()

        @dataclass
        class Results():
            Qs_inc_w:       float = 0.0
            inc_flux_max:   float = 0.0
            trigger:        float = 0.0
            cp_avg:         float = 0.0
            Ti:             float = 0.0
            To_lhs:         float = 0.0
            To_rhs:         float = 0.0
            Qfluid_W:       float = 0.0
            N_hel_used:     float = 0.0
            min_tube_LTEs:  float = 0.0
            offset_factor:  float = 0.0
            cut_Hstats:     float = 0.0
            LTEs:           float = 0.0
            dTs:            float = 0.0
            Tfs:            float = 0.0

            @dataclass
            class MassFlow():
                lhs: float = 3.0
                rhs: float = 0.0
            mdot: MassFlow = field(default_factory=MassFlow)
        self.res = Results()

if __name__=='__main__':

    # des_a = Prometheus()
    # des_a.hel.img.rel_areas = np.zeros((5, 4))
    # foo = des_a.hel.img.rel_areas

    # print(foo)

    columns = ['keys', 'widths', 'heights', 'rel_areas']
    image_df = pd.DataFrame(columns=columns)
    # image_df.loc[0:3, 'keys'] = [1, 20, 3, 100]
    image_df.loc[0, 'keys'] = 1
    image_df.loc[1, 'keys'] = 20
    image_df.loc[2, 'keys'] = 3
    image_df.loc[3, 'keys'] = 100

    print(image_df[columns[0]])

#EOF
