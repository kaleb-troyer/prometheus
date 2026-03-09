"""
created by jwenner on 10/20/25 to document the effects of thermal model disc. parameters on receiver power and LTE.
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
    # nzs         =np.arange(10,200,5)
    # nthetas     =np.arange(10,300,5)
    # nrs         =np.arange(29,30)
    # ntubes_sims   =[13,11,9,7,5,3,1]
    ntubes_sims   =[3,]
    for ntubes_sim in ntubes_sims:
        ## set up the full steady state analysis object
        model =steady_state_analysis_jwenn.SteadyStateAnalysis()

        #--- Read existing receiver design from file and/or update receiver design input parameters
        input_name ='../receivers/resolution_study_receiver'
        model.receiver.load_from_json(input_name)

        model.receiver.disc                     =settings.Discretization(5, 79, 50)   # standard is 5, 79, 50. number of r, theta, z nodes 
        model.receiver.options.wall_detail      ='2D' 
        model.receiver.options.calculate_stress =False   # Calculate elastic thermal stress distributions?
        model.receiver.flow_control_mode        =0             # 0 = Control each path independently
        model.receiver.ntubesim                 =ntubes_sim    # Number of tubes simulated per panel

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
        case_string =f'Qdes{model.receiver.Qdes}_optRes{model.receiver.n_flux_x}_W{model.receiver.D:.1f}_H{model.receiver.H:.2f}_resolution_study'
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
        thermal_results={}
        thermal_results['dTs']=dTs_results.tolist()
        thermal_results['Tfs']=Tfs_results.tolist()

        # with open(f'{output_string}_thermal_results.json', "w") as f:
        #         json.dump(thermal_results, f)
            
        ## create damage model interpolator
        mat_string  ='A230'
        dmg_inst    =damage_tool.damageTool(mat_string)

        ## get LTEs
        LTEs                            =dmg_inst.get_LTEs_w_SR_penalty(dTs_results.flatten(),Tfs_results.flatten(),Rs_results.flatten())
        min_panel_LTEs, min_tube_LTEs   =dmg_inst.calc_minimum_panel_LTEs(model.receiver, LTEs)

        ## save each tube's local minimum, the nz resolution used, and the Qfluid recorded
        results_dict    ={}
        results_dict['opt_res'] =model.receiver.n_flux_x
        results_dict['Qfluid']  =model.results.Qfluid
        results_dict['nz']      =model.receiver.disc.nz
        results_dict['ntheta']  =model.receiver.disc.ntheta
        results_dict['nr']      =model.receiver.disc.nr
        results_dict['ntubes_sim']=model.receiver.ntubesim
        results_dict['tsolve']  =t_solve
        # for i,min_tube_LTE in enumerate(min_tube_LTEs):
        #     tube_name               =f'tube{i}'
        #     results_dict[tube_name] =min_tube_LTE
        for i,min_panel_LTE in enumerate(min_panel_LTEs):
            panel_name               =f'panel{i}'
            results_dict[panel_name]=min_panel_LTE

        results_df  =pd.DataFrame(dict([(key, pd.Series(value)) for key, value in results_dict.items()]))
        # # use this section to save report for plotting
        # if os.path.exists('./reports/'+case_string+'_ntubes_sim_study.csv'):
        #     results_df.to_csv('./reports/'+case_string+'_ntubes_sim_study.csv', mode='a', index=False, header=False)
        # else:
        #     results_df.to_csv('./reports/'+case_string+'_ntubes_sim_study.csv', mode='a', index=False, header=True)
        # #


    ## plot the nz resolution effects at optical res. of 52
                # get the nz dataframe
    fontsize        =18
    # nz_results_df   =pd.read_csv('./reports/Qdes200_optRes74_W18.0_H15.00_axial_resolution_study_nz_study.csv')
    optres          =74
    nz_results_df   =pd.read_csv(f'./reports/Qdes200_optRes{optres}_W18.0_H15.00_axial_resolution_study_nz_study.csv')
    # nz_results_df   =pd.read_csv('./reports/Qdes200_optRes74_W18.0_H15.00_axial_resolution_study_ntheta_study.csv')
    fig,ax          =plt.subplots(tight_layout=True)
    ntubes          =model.receiver.Npanels
    # cmap = plt.get_cmap('gray', ntubes)
    cmap            =plt.cm.gray
    # loop through all rows in dataframe
    for i,string in enumerate(nz_results_df.columns):
        if 'tube' in string:
            ii =i-3           # this is the number of precolumns that don't have tube in them
            ax.scatter(nz_results_df['nz']/optres, nz_results_df[string]/nz_results_df[string].max(), color=cmap(ii/ntubes), marker='s', s=8 )
            ax.scatter(nz_results_df['nz']/optres, nz_results_df[string]/nz_results_df[string].max(), color=cmap(ii/ntubes), marker='s', s=8 )
            ax.plot(nz_results_df['nz']/optres, nz_results_df[string]/nz_results_df[string].max(), linestyle='--', color=cmap(ii/ntubes), linewidth=1 )
            ax.plot(nz_results_df['nz']/optres, nz_results_df[string]/nz_results_df[string].max(), linestyle='--', color=cmap(ii/ntubes), linewidth=1 )
        else:
            pass
    ax.set_xlabel('# axial thermal model nodes / # axial optical model nodes',fontsize=fontsize)
    ax.set_ylabel('normalized minimum tube lifetimes', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    # fig.savefig(f'imgs/axial_resolution_study_min_LTEs_{optres}opt_nz', dpi=300)
    plt.show()
    plt.close()

    fig,ax =plt.subplots(tight_layout=True)
    ax.plot(nz_results_df['nz']/optres, nz_results_df['Qfluid']/1e6, linewidth=1, linestyle='--')
    ax.scatter(nz_results_df['nz']/optres, nz_results_df['Qfluid']/1e6, marker='s', s=8)
    ax.set_xlabel('# axial thermal model nodes / # axial optical model nodes',fontsize=fontsize)
    ax.set_ylabel('absorbed thermal power (MWth)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    # fig.savefig(f'imgs/axial_resolution_study_Qfluid_{optres}opt_nz', dpi=300)
    plt.show()
    plt.close()
    ##
    
    
    
    ## plot the ntheta results at optical res. of 74
    optres          =52
    ntheta_results_df   =pd.read_csv(f'./reports/Qdes200_optRes{optres}_W18.0_H15.00_resolution_study_ntheta_study.csv')
    fig,ax          =plt.subplots(tight_layout=True)
    ntubes          =model.receiver.Npanels
    cmap            =plt.cm.gray
    # loop through all rows in dataframe
    for i,string in enumerate(ntheta_results_df.columns):
        if 'tube' in string:
            ii =i-3 if optres==74 else i-4          # this is the number of precolumns that don't have tube in them. changed midway through study
            ax.scatter(ntheta_results_df['ntheta'], ntheta_results_df[string]/ntheta_results_df[string].max(), color=cmap(ii/ntubes), marker='s', s=8 )
            ax.scatter(ntheta_results_df['ntheta'], ntheta_results_df[string]/ntheta_results_df[string].max(), color=cmap(ii/ntubes), marker='s', s=8 )
            ax.plot(ntheta_results_df['ntheta'], ntheta_results_df[string]/ntheta_results_df[string].max(), linestyle='--', color=cmap(ii/ntubes), linewidth=1 )
            ax.plot(ntheta_results_df['ntheta'], ntheta_results_df[string]/ntheta_results_df[string].max(), linestyle='--', color=cmap(ii/ntubes), linewidth=1 )
        else:
            pass
    ax.set_xlabel('# circumferential nodes',fontsize=fontsize)
    ax.set_ylabel('normalized minimum tube lifetimes', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    # fig.savefig(f'imgs/resolution_study_min_LTEs_{optres}Opt_ntheta', dpi=300)
    plt.show()
    plt.close()

    fig,ax =plt.subplots(tight_layout=True)
    ax.plot(ntheta_results_df['ntheta'], ntheta_results_df['Qfluid']/1e6, linewidth=1, linestyle='--')
    ax.scatter(ntheta_results_df['ntheta'], ntheta_results_df['Qfluid']/1e6, marker='s', s=8)
    ax.set_xlabel('# circumferential nodes',fontsize=fontsize)
    ax.set_ylabel('absorbed thermal power (MWth)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    # fig.savefig(f'imgs/resolution_study_Qfluid_{optres}Opt_ntheta', dpi=300)
    plt.show()
    plt.close()
    ##

    ## plot the nr results at optical res. of 74 and 52
    nr_results_df   =pd.read_csv(f'./reports/Qdes200_optRes74_W18.0_H15.00_resolution_study_nr_study.csv')
    fig,ax          =plt.subplots(tight_layout=True)
    opt_res         =nr_results_df['opt_res'][0]
    ntubes          =model.receiver.Npanels
    cmap            =plt.cm.gray
    # loop through all rows in dataframe
    for i,string in enumerate(nr_results_df.columns):
        if 'tube' in string:
            ii =i-6           # this is the number of precolumns that don't have tube in them
            ax.scatter(nr_results_df['nr'], nr_results_df[string]/nr_results_df[string].max(), color=cmap(ii/ntubes), marker='s', s=8 )
            ax.scatter(nr_results_df['nr'], nr_results_df[string]/nr_results_df[string].max(), color=cmap(ii/ntubes), marker='s', s=8 )
            ax.plot(nr_results_df['nr'], nr_results_df[string]/nr_results_df[string].max(), linestyle='--', color=cmap(ii/ntubes), linewidth=1 )
            ax.plot(nr_results_df['nr'], nr_results_df[string]/nr_results_df[string].max(), linestyle='--', color=cmap(ii/ntubes), linewidth=1 )
        else:
            pass
    ax.set_xlabel('# radial nodes',fontsize=fontsize)
    ax.set_ylabel('normalized minimum tube lifetimes', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    # fig.savefig(f'imgs/resolution_study_min_LTEs_{opt_res}Opt_nr', dpi=300)
    plt.show()
    plt.close()

    fig,ax =plt.subplots(tight_layout=True)
    ax.plot(nr_results_df['nr'], nr_results_df['Qfluid']/1e6, linewidth=1, linestyle='--')
    ax.scatter(nr_results_df['nr'], nr_results_df['Qfluid']/1e6, marker='s', s=8)
    ax.set_xlabel('# radial nodes',fontsize=fontsize)
    ax.set_ylabel('absorbed thermal power (MWth)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.yaxis.set_major_formatter("{x:.2f}") # Formats to 3 decimal places
    ax.xaxis.set_major_formatter("{x:.0f}") # Formats to 3 decimal places
    # fig.savefig(f'imgs/resolution_study_Qfluid_{opt_res}Opt_nr', dpi=300)
    plt.show()
    plt.close()

    ## plot the nr results at optical res. of 74 and 52
    ntubes_sim_results_df   =pd.read_csv(f'./reports/Qdes200_optRes74_W18.0_H15.00_resolution_study_ntubes_sim_study.csv')
    fig,ax          =plt.subplots(tight_layout=True)
    opt_res         =ntubes_sim_results_df['opt_res'][0]
    ntubes          =model.receiver.Npanels
    cmap            =plt.cm.gray
    # loop through all rows in dataframe
    for i,string in enumerate(ntubes_sim_results_df.columns):
        if 'panel' in string:
            ii =i-7           # this is the number of precolumns that don't have tube in them
            ax.scatter(ntubes_sim_results_df['ntubes_sim'], ntubes_sim_results_df[string]/ntubes_sim_results_df[string].max(), color=cmap(ii/ntubes), marker='s', s=8 )
            ax.scatter(ntubes_sim_results_df['ntubes_sim'], ntubes_sim_results_df[string]/ntubes_sim_results_df[string].max(), color=cmap(ii/ntubes), marker='s', s=8 )
            ax.plot(ntubes_sim_results_df['ntubes_sim'], ntubes_sim_results_df[string]/ntubes_sim_results_df[string].max(), linestyle='--', color=cmap(ii/ntubes), linewidth=1 )
            ax.plot(ntubes_sim_results_df['ntubes_sim'], ntubes_sim_results_df[string]/ntubes_sim_results_df[string].max(), linestyle='--', color=cmap(ii/ntubes), linewidth=1 )
        else:
            pass
    ax.set_xlabel('simulated tubes / panel',fontsize=fontsize)
    ax.set_ylabel('normalized minimum tube lifetimes', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    # fig.savefig(f'imgs/resolution_study_min_LTEs_{opt_res}opt_ntubes_sim', dpi=300)
    plt.show()
    plt.close()

    fig,ax =plt.subplots(tight_layout=True)
    ax.plot(ntubes_sim_results_df['ntubes_sim'], ntubes_sim_results_df['Qfluid']/1e6, linewidth=1, linestyle='--')
    ax.scatter(ntubes_sim_results_df['ntubes_sim'], ntubes_sim_results_df['Qfluid']/1e6, marker='s', s=8)
    ax.set_xlabel('simulated tubes / panel',fontsize=fontsize)
    ax.set_ylabel('absorbed thermal power (MWth)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.yaxis.set_major_formatter("{x:.2f}") # Formats to 3 decimal places
    ax.xaxis.set_major_formatter("{x:.0f}") # Formats to 3 decimal places
    # fig.savefig(f'imgs/resolution_study_Qfluid_{opt_res}opt_ntubes_sim', dpi=300)
    plt.show()
    plt.close()

    # make bar plots of current minimum panel LTEs. NOTE: uses the solved model results you request instead of from a csv.
    fig, ax =plt.subplots(tight_layout=True)
    # ax.grid(axis='y', which='both', alpha=0.2,zorder=1)
    npan    =model.receiver.Npanels
    panels  =np.arange(npan, dtype=int)
    for i,panel in enumerate(panels):
        min_tube_LTE_max=min_tube_LTEs[i].max()
        if i == panels.size-1:
            ax.scatter(panel*np.ones(int(ntubes_sim-2)), min_tube_LTEs[i][1:-1]/min_tube_LTE_max,marker='s', color='gray', label='inner')
            ax.scatter(panel,min_tube_LTEs[i][0]/min_tube_LTE_max,marker='o', color='r', label='left')
            ax.scatter(panel,min_tube_LTEs[i][-1]/min_tube_LTE_max,marker='x', color='b', label='right')
        else:
            ax.scatter(panel*np.ones(int(ntubes_sim-2)), min_tube_LTEs[i][1:-1]/min_tube_LTE_max, marker='s', color='gray')
            ax.scatter(panel,min_tube_LTEs[i][0]/min_tube_LTE_max, marker='o', color='r')
            ax.scatter(panel,min_tube_LTEs[i][-1]/min_tube_LTE_max, marker='x', color='b')
    ax.xaxis.set_major_locator(tck.MultipleLocator()) # thanks stack exchange!
    ax.set_xlabel('panel number', fontsize=fontsize)
    ax.set_ylabel('lifetime (yrs)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-1)
    # ax.set_xlim(-0.5,npan-0.25)
    ax.legend(ncols=3, fontsize=fontsize-4, columnspacing=0.1, handletextpad=0.05)
    fig.savefig(f'imgs/resolution_ntubes_sim_study_all_tube_LTEs_{ntubes_sim}per_panel',dpi=300)
    plt.show()
    plt.close()