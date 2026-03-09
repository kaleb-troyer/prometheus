
"""
created by jwenner on 10/19/25 to make informed flux profiles for given receiver sizes
"""
## append the native HALOS folder to path for importing
import sys
sys.path.append('..') # thermal model is in upper level folder
sys.path.append('aiming_informer/')
sys.path.append('../Chapter_3/damage_tool/')
sys.path.append('../Chapter_3/thermal_and_optical_tools/')
import helpers_thermal_model 
import pandas as pd
import numpy as np
import timeit
import informed_aiming
import matplotlib.pyplot as plt
import damage_tool
import tube_jwenner
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import json
import copy

if __name__ =='__main__': # very important if multiprocessing. Prevents subprocesses from being started
    # vvv uncomment to make ideal flux files
    ARs             =[0.83,]    # shortened list for debugging
    # ARs =[0.23, 0.26, 0.30, 0.34, 0.40, 0.47, 0.56, 0.68, 0.83, 1.05, 1.38, 1.88, 2.70]
    # ARs =[2.70,0.23]
    # ARs =[1.05]
    Qdes=220
    ## make a list of fluxgrid arrays for later program use
    fluxgrid_list   =[]
    for AR in ARs:
        ## if using offset, set here
        v_offset    =2          # offsets from the edge by 2m
        using_offset=False
        max_LTE     =False      # if true, program will find the highest possible lifetime contour to design with

        ## designate a cutoff to avoid stress reset region when building contour. Default is 270
        cutoff      =269

        ## set the receiver dimensions
        W_panel =1
        A_og    =15*18  # (m2) original receiver area 
        

        ## generate the ideal flux profile using desired lifetime of 80 years
        tube        =tube_jwenner.Tube()
        tube.OD     =0.0508             # (m)
        tube.twall  =0.00125         # (m)
        tube.options.is_adjacent_tubes  =False # prevents annoying message when initializing the tube
        tube.solar_abs                  =0.96
        tube.initialize()

        LTE_desired =30     # units in years

        ## make the model to get mass flow
        # receiver_filestring ='receivers/resolution_study_receiver'
        receiver_filestring =f'receivers/AR_study/billboard_{Qdes}Qdes_{AR:.2f}AR'

        model_lite  =helpers_thermal_model.setup_LWT_thermal_model(filestring=receiver_filestring,ntubes_sim=1)
        # calculate the height and width of the receiver using the aspect ratio
        W   =model_lite.D #np.sqrt(A_og/AR)
        H   =model_lite.H
        
        # model_lite.Npanels  =np.round(W/W_panel)
        print(f'number of receiver panels: {model_lite.Npanels}')
        
        # HALOS_res   =model_lite.Npanels*3   # the square resolution for solarpilot simulation, ideal flux grid, and thermal model input
        mflow       =np.sum(model_lite.operating_conditions.mass_flow)  # the setup_LWT_thermal_model automatically selects a mass flow rate based on the design power and outlet temp. range

        ## instantiate a damage case
        mat_string ='A230'
        dmg_inst =damage_tool.damageTool(mat_string=mat_string )   

        flowpath_config =f'{model_lite.npaths}_{model_lite.start_pt}'

        # plot and (optionally) save json of what the ideal fluxmap looks like
        filestring_fluxmap =f"aiming/{AR:.2f}AR_{A_og:.0f}area_ideal_fluxmap_with_offset.json"
        imgstring_fluxmap =f"{AR:.2f}AR_{A_og:.0f}area_ideal_fluxmap"

        if not using_offset:
            rec_flux_dict, Q_inc_rec, Areas, A_req =informed_aiming.generate_ideal_fluxmap(dmg_inst, tube, LTE_desired, mflow, W_panel, H, flowpath_config, cutoff=cutoff)
            informed_aiming.plot_ideal_fluxmap(rec_flux_dict, H, W, '2_ctr', imgstring_fluxmap)
            informed_aiming.save_ideal_fluxmap(rec_flux_dict, filestring_fluxmap)
        else:
            rec_flux_dict, Q_inc_rec, Areas, A_req =informed_aiming.generate_ideal_fluxmap_with_offset(dmg_inst, tube, LTE_desired, mflow, W_panel, H, v_offset, flowpath_config, cutoff=cutoff)
            # rec_flux_dict, Q_inc_rec, Areas, A_req =informed_aiming.generate_ideal_fluxmap_with_offset(dmg_inst, tube, 490, mflow, W_panel, H, v_offset, flowpath_config) # used this for a problem contour
            informed_aiming.plot_ideal_fluxmap_w_offset(rec_flux_dict, H, W, '2_ctr', imgstring_fluxmap)
            if max_LTE:
                rec_flux_dict, Q_inc_rec, Areas, Areqd   =informed_aiming.fully_utilize_receiver_maximize_lifetime(dmg_inst, tube, mflow, W_panel, W, H, v_offset, flowpath_config, cutoff=cutoff)
                informed_aiming.plot_ideal_fluxmap_w_offset(rec_flux_dict, H, W, '2_ctr', imgstring_fluxmap)


        # also note that the ideal flux grid is in units of W

        # need to check the width and raise error if the ideal fluxmap requires more panels than the receiver has
        if flowpath_config == '2_ctr':
            N_panels_half   =len(rec_flux_dict.keys())
            N_panels_ideal  =N_panels_half*2

        # # determine the y resolution
        y_res_ratio =0.27778    # (m/point)
        res_y       =int(np.round(H/y_res_ratio))
        # res_y         =N_panels_ideal


        # important! must give the build_ideal_fluxgrid function the fluxmap's width!!!
        if not using_offset:
            ideal_fluxgrid =informed_aiming.build_ideal_fluxgrid(filestring_fluxmap,res_y=res_y, H=H, W=W_panel*N_panels_ideal, flowpath_config=flowpath_config)
        else: 
            ideal_fluxgrid =informed_aiming.build_ideal_w_offset_fluxgrid(rec_flux_dict, res_y=res_y, H=H, W=W_panel*N_panels_ideal, flowpath_config=flowpath_config)

        if N_panels_ideal != np.round(W/W_panel):
            print('Error! The minimum panel number for receiver @ this height differs from that in receiver object. adapting to match thermal model')
            ideal_fluxgrid =informed_aiming.fit_grid_to_receiver(ideal_fluxgrid, model_lite, flowpath_config)
        
        informed_aiming.plot_ideal_fluxgrid(ideal_fluxgrid, H, W, flowpath_config, imgstring_fluxmap+'_demo_spt_method')

        ## save the ideal fluxgrid for copylot input

        x_res_ratio =18/54  # (m/point)
        res_x       =int(np.round(W/x_res_ratio))
        ideal_fluxgrid =helpers_thermal_model.increase_flux_resolution_blocked_custom(ideal_fluxgrid, res_x, res_y)
        print(f'resolution for AR{AR} is {ideal_fluxgrid.shape}')

        # save the flux grid as a csv
        aiming_df=pd.DataFrame(data=ideal_fluxgrid, index=None, columns=None, dtype=None, copy=None) # create empty dataframe 
        aiming_df = aiming_df.iloc[::-1]    # flip this to prepare for SPT input
        # # aiming_df.to_csv(f'aiming/ideal_fluxgrid_{model_lite.Qdes:.0f}MWth_{v_offset:.0f}Voffset_{W_panel:.1f}panel.csv',header=None, index=None)
        aiming_df.to_csv(f'aiming/ideal_fluxgrid_{model_lite.Qdes:.0f}MWth_{AR:.2f}AR_{A_og:.0f}area.csv',header=None, index=None) # this is the most recently used save line



        fluxgrid_list.append(ideal_fluxgrid)

    #     # save the flux grid as a csv
    #     aiming_df=pd.DataFrame(data=ideal_fluxgrid, index=None, columns=None, dtype=None, copy=None) # create empty dataframe 
    #     aiming_df = aiming_df.iloc[::-1]    # flip this to prepare for SPT input
    #     aiming_df.to_csv(f'{filestring_fluxmap}.csv',header=None, index=None)
    # end of for loop through AR list

    ## test to see if perfect execution of the flux grid yields expected LTEs

    # ARs =[0.23, 0.26, 0.30, 0.34, 0.40, 0.47, 0.56, 0.68, 0.83, 1.05, 1.38, 1.88, 2.70] # uncomment this line to run a select for-loop different than the one above
    # make damage class
    mat_string ='A230'
    dmg_inst =damage_tool.damageTool(mat_string=mat_string )  

    for index,AR in enumerate(ARs):
        ## read the aiming file in as a flux file
        flux_df     =pd.read_csv(f'aiming/ideal_fluxgrid_{Qdes}MWth_{AR:.2f}AR_270area.csv', header=None)
        flux_df     =flux_df.iloc[::-1]    # flip this to prepare for SPT input
        inc_flux_W=flux_df.values
        inc_flux=inc_flux_W # inc_flux will be converted to kW later

        # select the correct receiver file
        AR      =ARs[index]
        # flux_input  =fluxgrid_list[index]
        flux_input  =inc_flux
        receiver_filestring =f'receivers/AR_study/billboard_{Qdes}Qdes_{AR:.2f}AR'
        # receiver_filestring ='receivers/resolution_study_receiver'
        ideal_receiver  =helpers_thermal_model.setup_LWT_thermal_model(filestring=receiver_filestring,ntubes_sim=3)
        print(f"receiver width:{ideal_receiver.D:.2f}, height:{ideal_receiver.H:.2f}, number of panels:{ideal_receiver.Npanels}")
        # solve model
        flux_input =helpers_thermal_model.increase_flux_resolution_blocked_custom(flux_input, flux_input.shape[1]*9, flux_input.shape[0]*9)
        helpers_thermal_model.solve_LWT_thermal_model(ideal_receiver, flux_input/1e3)

        # get LTEs & plot
        dTs_LWT_ideal, Tfs_LWT_ideal, qabs_LWT_ideal, Rs_LWT_ideal  =helpers_thermal_model.get_thermal_results(ideal_receiver)
        LTEs_LWT_ideal                                              =dmg_inst.get_LTEs(dTs_LWT_ideal.flatten(),Tfs_LWT_ideal.flatten(),Rs_LWT_ideal.flatten())
        # picture where the points are on the damage map
        dmg_inst.plot_dmg_map(op_dTs=dTs_LWT_ideal,op_Tfs=Tfs_LWT_ideal) 

        print(f'minimum lifetime is {np.min(LTEs_LWT_ideal)}')
        savename ='default' #f'run_uniform_and_ideal_LTEs_for_Qdes{int(ideal_receiver.Qdes)}_ideal'
        helpers_thermal_model.plot_results(ideal_receiver, LTEs_LWT_ideal.reshape(dTs_LWT_ideal.shape), label_name='lifetimes (yrs)', savename=savename, vmin=0, vmax=100)