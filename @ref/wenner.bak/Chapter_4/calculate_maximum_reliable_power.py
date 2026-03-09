"""
created by jwenner on 9/3/2025
calculates the maximum power a given receiver can have for a given lifetime contour by increasing Qdes until Npanels_req > Npanels_actual
"""
## append the native HALOS folder to path for importing
import sys
sys.path.append('./aiming_informer/')
sys.path.append('./../Chapter_3/thermal_and_optical_tools/')
sys.path.append('./../Chapter_3/damage_tool/')
import helpers_thermal_model as LWT_therm_mod_helpers 
import pandas as pd
import numpy as np
import matplotlib.animation as animation
import timeit
import informed_aiming
import matplotlib.pyplot as plt
import damage_tool
import tube_jwenner
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == "__main__":
    receiver_filestring = 'receivers/billboard_test_mod'
    model_hvy  =LWT_therm_mod_helpers.setup_LWT_thermal_model(filestring=receiver_filestring,ntubes_sim=3)
    start   =100
    end     =400
    Qdesigns    =np.linspace(start, end, 300)
    Areas_required=[]
    LTE_desired =30

    tube        =tube_jwenner.Tube()
    tube.OD     =0.0508             # (m)
    tube.twall  =0.00125         # (m)
    tube.options.is_adjacent_tubes =False # prevents annoying error when initializing the tube
    tube.initialize()

    mat_string ='A230'
    dmg_inst =damage_tool.damageTool(mat_string) 
    i=0
    Areq_total=0
    model_hvy   =LWT_therm_mod_helpers.setup_LWT_thermal_model(filestring=receiver_filestring,ntubes_sim=3)
    nx =60
    ## use these sections to calculate the maximum reliable power
    # while Areq_total <= model_hvy.D*model_hvy.H:
    ##
    for Q in Qdesigns:
        ##
        # mflow       =(Qdesigns[i]*1e6)/model_hvy.HTFcpavg/(model_hvy.Tfout_design-model_hvy.Tfin_design)  # the setup_LWT_thermal_model automatically selects a mass flow rate based on the design power and outlet temp. range
        ##
        mflow       =(Q*1e6)/model_hvy.HTFcpavg/(model_hvy.Tfout_design-model_hvy.Tfin_design)  # the setup_LWT_thermal_model automatically selects a mass flow rate based on the design power and outlet temp. range
        W_panel     =model_hvy.D/model_hvy.Npanels # for now, the width is determined by the number of panels, which is set so that the velocity matches the reference velocity of the prelim study
        L_t         =model_hvy.tubes[0][0].length
        flowpath_config =f'{model_hvy.npaths}_{model_hvy.start_pt}'
        rec_flux_dict, Q_inc_rec, Areas, Areq_total =informed_aiming.generate_ideal_fluxmap(dmg_inst, tube, LTE_desired, mflow, W_panel, L_t, flowpath_config, False)
        Areas_required.append(Areq_total)



        nPanels_req =len(list(rec_flux_dict.keys() )*2 )          # use for determining max possible power
        ## get the number of panels, assuming 2_ctr configuration 
        # i+=1
        ##
        
        # ----- unfinished: calculate the pumping power to place maximum reliable power in context ----------------
        # ## apply the flux uniformly, solve the model, then calculate the pumping power
        # A_grid_pt =Areq_total/(nx*nx)
        # flux_profile =(Q/(nx*nx))/A_grid_pt*np.ones((nx,nx))
        # dTs, Tfs, qabs, Rs = LWT_therm_mod_helpers.solve_LWT_thermal_model(model_hvy, flux_profile)
        # Q_pump =LWT_therm_mod_helpers.get_pumping_power(model_hvy)
        # --------------------------------

    ## use this section to calculate the maximum reliable power
    # print(f'the highest power that can be accomplished in this configuration is: {Qdesigns[i]} MWth')
    # informed_aiming.plot_ideal_fluxmap(rec_flux_dict, model_hvy.H, W_panel*(nPanels_req), flowpath_config)
    ##

    fontsize =14
    fig,ax =plt.subplots()
    ax.plot(Qdesigns, Areas_required)
    # ## add a m=1 linear line to show difference
    # ylinears = 1*np.array(Qdesigns-Qdesigns[0]) + Areas_required[0]
    # ax.plot(Qdesigns, ylinears)
    # ##
    ax.set_ylabel('required area (m$^2$)',fontsize=fontsize)
    ax.set_xlabel('design power (MWth)',fontsize=fontsize)
    ax.set_xlim(start, end)
    ax.grid(True)
    fig.savefig('imgs/default_save_req_area_vs_design_power', dpi=300)
    plt.show()
    plt.close()

    