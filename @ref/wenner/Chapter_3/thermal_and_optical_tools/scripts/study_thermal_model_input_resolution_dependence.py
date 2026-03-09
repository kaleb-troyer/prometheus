"""
created by jwenner on 10/26/25 to document the effects of thermal model disc. parameters on receiver power and LTE.
"""
## import all the model stuff
import sys
sys.path.append('..') # thermal model is in upper level folder
sys.path.append('../../damage_tool/')
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

if __name__ == '__main__':



    block_factor                            =9 # set this for each run to be consistent with what we are using in thermal model
    ## set up the full steady state analysis object
    model =steady_state_analysis_jwenn.SteadyStateAnalysis()

    #--- Read existing receiver design from file and/or update receiver design input parameters
    input_name ='../receivers/resolution_study_receiver'
    model.receiver.load_from_json(input_name)

    model.receiver.disc                     =settings.Discretization(5, 79, 50)   # standard is 5, 79, 50. number of r, theta, z nodes 
    model.receiver.options.wall_detail      ='2D' 
    model.receiver.options.calculate_stress =False   # Calculate elastic thermal stress distributions?
    model.receiver.flow_control_mode        =0             # 0 = Control each path independently
    model.receiver.ntubesim                 =3    # Number of tubes simulated per panel

    model.analysis_mode         ='design_point' #'design_day'  ,- was using this 1/27/25, og comment -># 'design_point', 'design_day', 'three_day' (summer solstice, equinox, winter solstice), 'selected_days', 'user_defined' # was using "selected_days"
    model.analysis_days         =[172] # can manually set this but analysis_mode will default set this as well
    substeps                    =0.5                    # number of model substeps per hour
    model.delta_hour            =1/substeps                # Time spacing between simulated time points (hr)
    model.is_half_day           =False           # Simulate only half of the day?
    model.dni_setting           ='user-defined-noon'  
    model.user_defined_dni      ={172:950, 264:980, 355:930} # User-defined DNI at solar noon on each simulated day.  Used in conjunction with clearsky-DNI to set DNI per time point
    model.ambient_setting       ='user-defined'
    model.user_defined_Tamb     =25+273.15    # Constant user-defined ambient T (K)
    model.user_defined_RH       =25             # Constant user-defined relative humidity
    model.user_defined_vwind10  =0         # Constant user-defined wind speed (m/s at 10m height)

    ## configure to run from a saved flux profile
    model.check_for_existing_flux_distributions =True  # Check if flux distributions for this configuration have already been saved
    model.save_flux_distributions               =False
    model.SF.saved_flux_direc                   ='./flux_sims/'  # Directory to store flux distributions

    ## run for single timepoint
    # case_string =f'Qdes{model.receiver.Qdes}_yRes{model.receiver.n_flux_y}_W{model.receiver.D:.1f}_H{model.receiver.H:.2f}_resolution_study'
    case_string =f'Qdes{model.receiver.Qdes}_W{model.receiver.D:.1f}_H{model.receiver.H:.2f}_narrower_panel_blocked_resolution_study'
    
    t_start     =timeit.default_timer()
    model.solve(verbosity=1)
    t_end       =timeit.default_timer()
    
    t_solve     =t_end-t_start

    # steady_state_plots.plot_profiles_in_path(model, day =172, hour_offset=0, use_all_tubes=False, savename=None)

    ## post-process to calculate receiver power and thermal operating points
    # # make a results json that contains overall results including Qfluid
    # d=0
    # output_string='./reports/'+case_string+'_report'
    # helpers_thermal_model.make_results_file(input_name,output_string,model,d)

    # make a second results json containing all the necessary LTE information
    dTs_results, Tfs_results, qabs_results, Rs_results  =helpers_thermal_model.get_thermal_results(model.receiver)

    ## use this code to save thermal points if desired
    # thermal_results         ={}
    # thermal_results['dTs']  =dTs_results.tolist()
    # thermal_results['Tfs']  =Tfs_results.tolist()
    # with open(f'{output_string}_thermal_results.json', "w") as f:
    #         json.dump(thermal_results, f)
    ##

    ## create damage model interpolator
    mat_string  ='A230'
    dmg_inst    =damage_tool.damageTool(mat_string)

    ## get LTEs
    LTEs                            =dmg_inst.get_LTEs_w_penalties(dTs_results.flatten(),Tfs_results.flatten(),Rs_results.flatten())
    min_panel_LTEs, min_tube_LTEs   =dmg_inst.calc_minimum_panel_LTEs(model.receiver, LTEs)

    ### save results of interest by setting up a dictionary
    results_dict                ={}
    results_dict['n_flux_x']    =model.receiver.n_flux_x
    results_dict['n_flux_y']    =model.receiver.n_flux_y
    results_dict['block']       =block_factor
    results_dict['Qfluid']      =model.results.Qfluid
    results_dict['nz']          =model.receiver.disc.nz
    results_dict['ntheta']      =model.receiver.disc.ntheta
    results_dict['nr']          =model.receiver.disc.nr
    results_dict['ntbs_sim']    =model.receiver.ntubesim
    results_dict['tsolve']      =t_solve

    ## save minimum tube lifetimes
    for i,min_tube_LTE in enumerate( min_tube_LTEs.flatten() ):
        tube_name               =f'tube{i}'
        results_dict[tube_name] =min_tube_LTE
    ##

    ## save minimum panel lifetimes
    # for i,min_panel_LTE in enumerate(min_panel_LTEs):
    #     panel_name               =f'panel{i}'
    #     results_dict[panel_name]=min_panel_LTE
    ##

    ###

    results_df  =pd.DataFrame(dict([(key, pd.Series(value)) for key, value in results_dict.items()]))

    # use this section to save report for plotting
    if os.path.exists('./reports/'+case_string+'_block_res_study.csv'):
        results_df.to_csv('./reports/'+case_string+'_block_res_study.csv', mode='a', index=False, header=False)
    else:
        results_df.to_csv('./reports/'+case_string+'_block_res_study.csv', mode='a', index=False, header=True)
    #

    ## plot the minimum tube lifetimes 
    fontsize        =18

    # n_flux_y        =74
    # results_df      =pd.read_csv(f'./reports/Qdes200_yRes{n_flux_y}_W18.0_H15.00_resolution_study_gap_res_study.csv')
    results_df      =pd.read_csv(f'./reports/Qdes200_W18.0_H15.00_narrower_panel_blocked_resolution_study_block_res_study.csv')
    # results_df      =pd.read_csv(f'./reports/Qdes200_W16.0_H15.00_blocked_resolution_study_block_res_study.csv')
    fig,ax          =plt.subplots(tight_layout=True)
    ntubes          =model.receiver.Npanels*model.receiver.ntubesim
    cmap            =plt.cm.gray

    # plot each tube's LTEs
    for i,string in enumerate(results_df.columns):
        if 'tube' in string:
            ii      =i-9           # this is the number of precolumns that don't have tube in them
            x_axis  =results_df['block']*model.receiver.n_flux_x/model.receiver.D # Npanels is assumed to always be equal to the fluxgrid dimension
            # ax.scatter(x_axis, results_df[string]/results_df[string].max(), color=cmap(ii/ntubes), marker='s', s=8 )
            ax.scatter(x_axis, results_df[string]/results_df[string].max(), color=cmap(ii/ntubes), marker='s', s=8 )
            ax.plot(x_axis, results_df[string]/results_df[string].max(), linestyle='--', color=cmap(ii/ntubes), linewidth=1 )
        else:
            pass
    ax.set_xlabel('# horizontal flux grid points / m',fontsize=fontsize)
    ax.set_ylabel('normalized minimum tube lifetimes', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    # fig.savefig(f'imgs/optical_res_blocking_impact_{model.receiver.D:.0f}m_receiver_narrower_panels', dpi=300)
    plt.show()
    plt.close()

    # fig,ax =plt.subplots(tight_layout=True)
    # ax.plot(nz_results_df['nz']/optres, nz_results_df['Qfluid']/1e6, linewidth=1, linestyle='--')
    # ax.scatter(nz_results_df['nz']/optres, nz_results_df['Qfluid']/1e6, marker='s', s=8)
    # ax.set_xlabel('# axial thermal model nodes / # axial optical model nodes',fontsize=fontsize)
    # ax.set_ylabel('absorbed thermal power (MWth)', fontsize=fontsize)
    # ax.tick_params(labelsize=fontsize)
    # # fig.savefig(f'imgs/axial_resolution_study_Qfluid_{optres}opt_nz', dpi=300)
    # plt.show()
    # plt.close()
    # ##