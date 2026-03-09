"""
Created by jwenner on 11/17/2025 to evaluate the annual LCOH for a range of aspect ratios
"""
## solve the thermal model at each of the 4 seasons

## calculate the total heating and pumping powers for every hour of the year

## calculate the total number of heliostats and required field

## calculate LCOH


## import stuff
import sys
sys.path.append('../Chapter_3/thermal_and_optical_tools/') # thermal model is in upper level folder
sys.path.append('../Chapter_3/damage_tool/') # damage tool
sys.path.append('./cost_model/')
import os
import steady_state_analysis_jwenn
import steady_state_plots
import settings
import timeit
import numpy as np
import matplotlib.pyplot as plt
import tube_jwenner
import numpy as np
import pandas as pd
import helpers_thermal_model
import json
import damage_tool
import matplotlib.ticker as tck
from scipy.optimize import curve_fit
import tower_model
from mpl_toolkits.axes_grid1 import make_axes_locatable


## adapted from helpers_thermal_model
def plot_results(W, H, Npanels, ntubesim, nz, results, label_name, savename='default_receiver_heatmap', vmin=0, vmax=1):
    """
    makes a heatmap using imshow, based on receiver width and height. Resolution is dependent on number of tubes/panel
    --
    results - a nz x npanels x ntubes/panel size array of whatever result you want to plot
    """
    cbr_fontsize =12
    fig,ax=plt.subplots(tight_layout=True)
    if label_name == 'lifetimes (yrs)' or label_name == 'log10(lifetimes (yrs))':
        cmap_choice ='plasma_r'
    else: 
        cmap_choice ='inferno'
    im = ax.imshow(results.reshape(nz,Npanels*ntubesim), extent=[-W/2, W/2, 0, H], vmin=vmin, vmax=vmax, cmap=cmap_choice )
    ax.set_xlabel('x position (m)',fontsize=cbr_fontsize+2)
    ax.set_ylabel('y position (m)',fontsize=cbr_fontsize+2)
    divider = make_axes_locatable(ax)  
    cax = divider.append_axes("right", size="2%", pad=0.08)     # thanks stack exchange!
    cbr = fig.colorbar(im, cax=cax)
    cbr.set_label(label=label_name, size=cbr_fontsize)
    cbr.ax.tick_params(labelsize=cbr_fontsize)
    fig.savefig(f'imgs/{savename}',dpi=300)
    plt.show()
    plt.close(fig)
    return 
##

if __name__ == '__main__':
    # ARs =[0.23, 0.26, 0.30, 0.34, 0.40, 0.47, 0.56, 0.68, 0.83, 1.05, 1.38, 1.88, 2.70]
    ARs =[0.23, 0.26, 0.30, 0.34, 0.40, 0.47, 0.56, 0.68, 0.83]
    # ARs =[1.38, 1.88, 2.70]
    # ARs =[0.23, 2.70]
    # # ARs =[0.83,]
    ## initialize the damage model
    matl    ='A230'
    dmg_obj =damage_tool.damageTool(mat_string=matl, interp_mode='LNDI')    # set interpolation mode to LNDI - improves accuracy and performance

    ## set design lifetime
    N_life =30  # (yrs)
    
    
    ## ------ start loop: series of aspect ratios ------ ##
    t_start =timeit.default_timer()
    for AR in ARs:
        ## set up the full steady state analysis object
        model =steady_state_analysis_jwenn.SteadyStateAnalysis()

        input_name =f'receivers/AR_study/billboard_220Qdes_{AR:.2f}AR'
        model.receiver.load_from_json(input_name)

        model.receiver.disc                     =settings.Discretization(5, 99, model.receiver.n_flux_y)   # standard is 5, 79, 50. number of r, theta, z nodes 
        model.receiver.options.wall_detail      ='2D' 
        model.receiver.options.calculate_stress =False   # Calculate elastic thermal stress distributions?
        model.receiver.flow_control_mode        =0             # 0 = Control each path independently
        model.receiver.ntubesim                 =3    # Number of tubes simulated per panel

        model.analysis_mode         ='selected_days' #'design_day'  ,- was using this 1/27/25, og comment -># 'design_point', 'design_day', 'three_day' (summer solstice, equinox, winter solstice), 'selected_days', 'user_defined' # was using "selected_days"
        model.analysis_days         =[172] # can manually set this but analysis_mode will default set this as well
        substeps                    =1                      # number of model substeps per hour
        model.delta_hour            =1/substeps                # Time spacing between simulated time points (hr)
        model.is_half_day           =False           # Simulate only half of the day?
        model.dni_setting           ='user-defined-noon'  
        model.user_defined_dni      ={172:950, 264:980, 355:930} # User-defined DNI at solar noon on each simulated day.  Used in conjunction with clearsky-DNI to set DNI per time point
        model.ambient_setting       ='user-defined'
        model.user_defined_Tamb     =25+273.15    # Constant user-defined ambient T (K)
        model.user_defined_RH       =25             # Constant user-defined relative humidity
        model.user_defined_vwind10  =0         # Constant user-defined wind speed (m/s at 10m height)

        ## configure to run from a saved flux profile
        model.check_for_existing_flux_distributions =False  # Check if flux distributions for this configuration have already been saved
        model.save_flux_distributions               =False
        model.SF.saved_flux_direc                   ='./flux_sims/'  # Directory to store flux distributions

        # # uncomment if using uniform aiming instead
        # print('using uniform aiming scheme!!!')
        # model.receiver.use_aiming_scheme    =0
        # model.receiver.aiming_file          ='None'
        # #

        ###
        case_string =f'Qdes{model.receiver.Qdes}_W{model.receiver.D:.1f}_H{model.receiver.H:.2f}_AR{AR:.2f}_LCOH_results'
        ###

        ## get the seasonal DNI reference
        with open('seasonal_DNI',) as f:
            DNI_seasons=json.load(f)
        t_samples   =np.arange(0,24)+0.5

        ## include the tower pressure when caclulating pumping power
        model.receiver.include_tower_in_Qpump =True

        ### get summer performance
        model.analysis_days                     =[172] # 172 can manually set this but analysis_mode will default set this as well
        model.solve(verbosity=1)
        DNIs_summer_day                         =np.array(DNI_seasons['summer'])
        Qfluids_summer_day, Qpumps_summer_day   =helpers_thermal_model.interp_Qs(model, t_samples, DNIs_summer_day)  # interpolate Qs from the simulated timepoints on day 172
        days_summer                             =31+30+31
        Qfluids_summer  =np.tile(Qfluids_summer_day.squeeze(), days_summer)   # assume every day looks like the average DNI day we interpolated the performance curves at
        Qpumps_summer   =np.tile(Qpumps_summer_day.squeeze(), days_summer)
        eta_opt_summer  =model.results.eta_field.tolist()

        ## get fall performance
        model.analysis_days                     =[264] # can manually set this but analysis_mode will default set this as well
        model.solve(verbosity=1)
        DNIs_fall_day                           =np.array(DNI_seasons['fall'])
        Qfluids_fall_day, Qpumps_fall_day       =helpers_thermal_model.interp_Qs(model, t_samples, DNIs_fall_day)
        days_fall                               =31+30+31
        Qfluids_fall                            =np.tile(Qfluids_fall_day.squeeze(), days_fall)
        Qpumps_fall                             =np.tile(Qpumps_fall_day.squeeze(), days_fall)

        ## get spring performance. No need to resimulate because we assume equinox
        DNIs_spring_day                         =np.array(DNI_seasons['spring'])
        Qfluids_spring_day, Qpumps_spring_day   =helpers_thermal_model.interp_Qs(model, t_samples, DNIs_spring_day)
        days_spring                             =28+31+30
        Qfluids_spring                          =np.tile(Qfluids_spring_day.squeeze(), days_spring)
        Qpumps_spring                           =np.tile(Qpumps_spring_day.squeeze(), days_spring)
        eta_opt_fall                            =model.results.eta_field.tolist()

        ## get winter performance
        model.analysis_days                     =[355] # can manually set this but analysis_mode will default set this as well
        model.solve(verbosity=1)
        DNIs_winter_day                         =np.array(DNI_seasons['winter'])
        Qfluids_winter_day, Qpumps_winter_day   =helpers_thermal_model.interp_Qs(model, t_samples, DNIs_winter_day)
        days_winter                             =31+30+31
        Qfluids_winter                          =np.tile(Qfluids_winter_day.squeeze(), days_winter)
        Qpumps_winter                           =np.tile(Qpumps_winter_day.squeeze(), days_winter)
        eta_opt_winter                          =model.results.eta_field.tolist()

        ## combine all seasons power to HTF and required pump power
        Qfluids_year =np.concatenate((Qfluids_spring, Qfluids_summer, Qfluids_fall, Qfluids_winter))
        Qpumps_year  =np.concatenate((Qpumps_spring, Qpumps_summer, Qpumps_fall, Qpumps_winter))*model.receiver.expected_cycle_eff
        P_pumps_year =Qpumps_year*model.receiver.expected_cycle_eff # Qpump from model is already divided by expected eta

        ## determine heliostat area
        n_hstats_used   =pd.read_csv(model.receiver.layout_file).index.max()+1-model.SF.Ndisable    # get total number of mirrors, subtract the number of disabled heliostats
        A_sf            =n_hstats_used*144  # assuming heliostat size is 144 m2

        ## determine the required land
        R_sf_land       =7.54    # ratio of required land to the total required heliostat area
        A_land          =R_sf_land*A_sf
        ## ^^^ comment all of this out if you just want to plot some lifetimes

        ## determine number of required panel replacements by solving the model at design point
        model.analysis_mode = 'design_point' #'design_day'  ,- was using this 1/27/25, og comment -># 'design_point', 'design_day', 'three_day' (summer solstice, equinox, winter solstice), 'selected_days', 'user_defined' # was using "selected_days"
        model.analysis_days = [172] # can manually set this but analysis_mode will default set this as well
        model.solve(verbosity=1)

        ## extract thermal points for damage tool
        dTs_design, Tfs_design, qabs_design, Rs_design =helpers_thermal_model.get_thermal_results(model.receiver)

        ## calculate lifetimes
        LTEs                            =dmg_obj.get_LTEs(dTs_design.flatten(), Tfs_design.flatten(), Rs_design.flatten())
        min_panel_LTEs, min_tube_LTEs   =dmg_obj.calc_minimum_panel_LTEs(model.receiver, LTEs)
        print('minimum panel LTE is:',min_panel_LTEs.min())

        # # ## show lifetime profiles and op points if desired
        # dmg_obj.plot_dmg_map(include_ratios=False, op_dTs=dTs_design.flatten(), op_Tfs=Tfs_design.flatten(), savename=f'thermal_points_dmg_map_AR_uni.png' )
        # plot_results(model.receiver.D, model.receiver.H, model.receiver.Npanels, model.receiver.ntubesim, model.receiver.disc.nz, LTEs, 'lifetimes (yrs)', savename=f'LTE_heatmap_{AR:.2f}.png', vmin=None, vmax=100) # use for A617. if you specifically want lifetimes, use 'lifetimes (yrs)' or 'log10(lifetimes (yrs))'


        ## calculate the number of replacements
        panel_replacements =np.floor(N_life/min_panel_LTEs)
        N_repl             =np.sum(panel_replacements)
        # dmg_obj.make_contour_function_from_interpolator(LTE_desired=150, cutoff=270, show_LTE_ctr=True)

        # ### test: compare results from uniform study in j: drive
        # Qdes=200
        # npan=18
        # filestring_baseline  =f'reports/Qdes200_npan18_W18.0_H15.0_uniform_report_thermal_results_SPT.json'
        # dTs_uniform          =helpers_thermal_model.json_to_np(filestring_baseline, "dTs")
        # Tfs_uniform          =helpers_thermal_model.json_to_np(filestring_baseline, "Tfs")
        # LTEs_uniform                                        =dmg_obj.get_LTEs(dTs_uniform.flatten(),Tfs_uniform.flatten(),0.5*np.ones(Tfs_uniform.size))
        # min_panel_LTEs_uniform, min_tube_LTEs_uniform       =dmg_obj.calc_minimum_panel_LTEs_simple_inputs(ntubes_sim = dTs_uniform.shape[2], axial_nodes = dTs_uniform.shape[0], Npanels = npan, LTEs = LTEs_uniform)
        # print('stop')
        # ###

        ## estimate LCOH with all information 
        times           =np.ones(8760)*3600 # that many hours in a year
        rec_cost_model  =tower_model.costModelFR(Qdes=model.receiver.Qdes*1e6, Htow=model.receiver.Htower, Hrec=model.receiver.H, Wrec=model.receiver.D, Wpanel=model.receiver.D/model.receiver.Npanels, D_o=50.8e-3, th=1.25e-3, material='A230', A_Hstats=A_sf, 
                            A_land=A_land, N_life=N_life, N_repl=N_repl, Qdot_HTFs=Qfluids_year.flatten(), times=times, P_el_pumps=P_pumps_year.flatten(), eta_PBII=0.45)
        LCOH            =rec_cost_model.calc_LCOH()*1e6*3600 # convert from euros/W-s to euros/MWh 

        ### log results
        ## make new dictionary with things to add
        results_dict = {}
        results_dict['Qfluids_yearly']  =Qfluids_year.flatten().tolist()
        results_dict['P_pumps_yearly']  =P_pumps_year.flatten().tolist()
        results_dict['Qpumps_yearly']   =Qpumps_year.flatten().tolist()
        results_dict['assume_cycle_eta']=model.receiver.expected_cycle_eff
        results_dict['N_replaced']      =N_repl
        results_dict['summer_opt_eff']  =eta_opt_summer
        results_dict['fall_opt_eff']    =eta_opt_fall
        results_dict['winter_opt_eff']  =eta_opt_winter
        results_dict['n_hstats_used']   =n_hstats_used
        results_dict['LCOH']            =LCOH
        results_dict['min_panel_LTEs']  =min_panel_LTEs.flatten().tolist()
        expected_cycle_eff              =model.receiver.expected_cycle_eff

        ## get all inputs from json input file
        inputName = input_name + '.json'
        outputName= f'reports/{case_string}.json'

        with open(inputName,) as f:
            report_dict=json.load(f)
        ## combine dictionaries
        report_dict.update(results_dict)
        ## put both dictionaries in a new file
        with open(outputName, "w") as f:
            json.dump(report_dict, f)
            
        del(model)
    t_end   =timeit.default_timer()
        
    ## ------ end aspect ratio loop ------ ##
    print('-'*10)
    print(f'total execution time is: {t_end-t_start}')
    print('-'*10)
    print(f'the calculated LCOH for the final run is {LCOH} euros/MWh')
    print(f'annual HTF energy is {(Qfluids_year.sum()/1e9):.2f} GWh')
    print(f'annual HTF pumping energy, assuming {expected_cycle_eff} cycle eff, is {(Qpumps_year.sum())/1e9:.2f} GWh')
    print('-'*10)