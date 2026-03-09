"""
created by jwenner on 9/4/25 to compare aiming methods in terms of LTEs achieved for various powers
"""
import numpy as np
import sys
sys.path.append('../Chapter_3/thermal_and_optical_tools/') # thermal model is in upper level folder
sys.path.append('../Chapter_3/damage_tool/') # damage tool
sys.path.append('./cost_model/')
# import json
import helpers_thermal_model as helpers
# import LWT_therm_mod_helpers 
import pandas as pd
import numpy as np
import matplotlib.animation as animation
import timeit
import aiming_informer
# import inputs
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
# import sp_module
import damage_tool
import tube_jwenner
# import helpers_heuristic
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


if __name__ == "__main__":
    N_totals                =[1868, 2206, 2549, 2925, 3371]    # total heliostats in each layout file
    Qdess                   =[140, 160, 180, 200, 220]          # corresponding design power

    ## for running single case
    # Qdess                   =[180,]
    # N_totals                =[2549,]
    ##

    ### setup all required thermal models
    receiver_filestring = 'receivers/billboard_comparison_study_SPCS25'
    model_hvy   = helpers.setup_LWT_thermal_model(filestring=receiver_filestring,ntubes_sim=3)
 
    ## for SPT results: import the json, convert to an array of dTs and Tfs
    npan                =18

    mat_string ='A230'
    dmg_inst    =damage_tool.damageTool(mat_string, interp_mode='LNDI') 
    # dmg_inst.R  =0.5
    dmg_inst.make_contour_function_from_interpolator(LTE_desired=30, cutoff=270)

    ## for all available results, plot every minimum panel lifetime
    fig2,ax2 =plt.subplots()
    fontsize =14

    ## start a list of global receiver minimums for each case so we can plot it later
    uniform_rec_mins =[]
    stage1pt5_rec_mins  =[]
    stage2_rec_mins     =[]

    for Qdes in Qdess:
        ## create a damage tool instance
        ## instantiate a damage case

        ## for SPT results: import the json, convert to an array of dTs and Tfs
        # filestring_baseline  =f'heuristic_comparison_study/Qdes{Qdes}nom_npan{npan}_W18.0_H15.0_stage_uniform_report_thermal_results.json'
        filestring_baseline  =f'reports/heuristic_comparison_study/Qdes{Qdes}_npan{npan}_W18.0_H15.0_uniform_report_thermal_results.json'
        dTs_uniform          =helpers.json_to_np(filestring_baseline, "dTs")
        Tfs_uniform          =helpers.json_to_np(filestring_baseline, "Tfs")
        ## calculate LTE using damage tool
        # LTEs_uniform                                        =dmg_inst.get_LTEs_w_SR_penalty(dTs_uniform.flatten(),Tfs_uniform.flatten(),0.5*np.ones(Tfs_uniform.size))
        LTEs_uniform                                        =dmg_inst.get_LTEs(dTs_uniform.flatten(),Tfs_uniform.flatten(),0.5*np.ones(Tfs_uniform.size))
        min_panel_LTEs_uniform, min_tube_LTEs_uniform       =dmg_inst.calc_minimum_panel_LTEs_simple_inputs(ntubes_sim = dTs_uniform.shape[2], axial_nodes = dTs_uniform.shape[0], Npanels = npan, LTEs = LTEs_uniform)
        print(f'minimum panel LTEs for uniform flux are: \n {min_panel_LTEs_uniform}')


        ## for SPT results: import the json, convert to an array of dTs and Tfs
        filestring_stage1pt5  =f'reports/heuristic_comparison_study/Qdes{Qdes}_npan{npan}_W18.0_H15.0_stage_1pt5_80yr_report_thermal_results.json'
        # filestring_stage1pt5  =f'heuristic_comparison_study/Qdes{Qdes}nom_npan{npan}_W18.0_H15.0_stage_1pt5_report_thermal_results.json'
        dTs_stage1pt5         =helpers.json_to_np(filestring_stage1pt5, "dTs")
        Tfs_stage1pt5         =helpers.json_to_np(filestring_stage1pt5, "Tfs")
        ## calculate LTE using damage tool
        # LTEs_stage1pt5                                      =dmg_inst.get_LTEs_w_SR_penalty(dTs_stage1pt5.flatten(),Tfs_stage1pt5.flatten(),0.5*np.ones(Tfs_stage1pt5.size))
        LTEs_stage1pt5                                      =dmg_inst.get_LTEs(dTs_stage1pt5.flatten(),Tfs_stage1pt5.flatten(),0.5*np.ones(Tfs_stage1pt5.size))
        min_panel_LTEs_stage1pt5, min_tube_LTEs_stage1pt5   =dmg_inst.calc_minimum_panel_LTEs_simple_inputs(ntubes_sim = dTs_stage1pt5.shape[2], axial_nodes = dTs_stage1pt5.shape[0], Npanels = npan, LTEs = LTEs_stage1pt5)
        print(f'minimum panel LTEs for stage 1.5 are: \n {min_panel_LTEs_stage1pt5}')

        # ## get heuristic results
        filestring_stage2   =f'reports/heuristic_comparison_study/compare_3D_heuristic_{Qdes}_results.json'
        # LTEs_stage2         =helpers.json_to_np(filestring_stage2, "LTEs")
        dTs_stage2          =helpers.json_to_np(filestring_stage2, "dTs")
        Tfs_stage2          =helpers.json_to_np(filestring_stage2, "Tfs")
        LTEs_stage2         =dmg_inst.get_LTEs(dTs_stage2.flatten(),Tfs_stage2.flatten(),0.5*np.ones(Tfs_stage2.size))

        min_panel_LTEs_stage2, min_tube_LTEs_stage2   =dmg_inst.calc_minimum_panel_LTEs_simple_inputs(ntubes_sim = dTs_stage2.shape[2], axial_nodes = dTs_stage2.shape[0], Npanels = npan, LTEs = LTEs_stage2) 
        print(f'minimum panel LTEs for stage 2 are: \n {min_panel_LTEs_stage2}')
        # # print(f'minimum tube LTEs for stage 1 are: \n {min_tube_LTEs_stage2}')

        # ## plot thermal points and overlay the damage contour
        # Tf_ctrs =np.linspace(290,565)
        # dT_ctrs =dmg_inst.dT_function(Tf_ctrs)

        # fig, ax =plt.subplots()
        # size=1
        # # ax.scatter(dTs_uniform.flatten(), Tfs_uniform.flatten(),label='Uniform',s=size,color='r')
        # ax.scatter(dTs_stage1pt5.flatten(), Tfs_stage1pt5.flatten(),label='IA-SPT',s=size,color='k')
        # ax.scatter(dTs_stage2.flatten(), Tfs_stage2.flatten(),label='IA-heuristic',s=size)
        # ax.plot(dT_ctrs, Tf_ctrs, label='30 yr contour', linestyle='--', color='gray')
        # ax.set_xlabel('total temperature difference (C)',fontsize=fontsize)
        # ax.set_ylabel('fluid temperature (C)', fontsize=fontsize)
        # ax.legend(fontsize=fontsize-2)
        # ax.tick_params(labelsize=fontsize-2)
        # fig.savefig(f'imgs/default_thermal_points_Qdes{Qdes}',dpi=300)
        # plt.show()
        # plt.close(fig)
        # ##

        # append the receiver minimum lifetime to lists
        uniform_rec_mins.append(np.min(min_panel_LTEs_uniform))
        stage1pt5_rec_mins.append(np.min(min_panel_LTEs_stage1pt5))
        stage2_rec_mins.append(np.min(min_panel_LTEs_stage2))

        # fig, ax =plt.subplots()
        # ax.grid(axis='y', which='both', alpha=0.2,zorder=1)
        # size=4
        # panels =np.arange(npan, dtype=int)
        # boxwidth=0.25

        # # colors were tan, olivedrab, and plum but changed at request of advisors
        # ax.bar(panels, min_panel_LTEs_stage1pt5,label='IA-SPT',width=boxwidth, zorder=2, edgecolor='k', color='k')
        # ax.bar(panels+boxwidth, min_panel_LTEs_stage2,label='IA-heuristic',width=boxwidth, zorder=2, edgecolor='k', color='b')
        # ax.bar(panels+2*boxwidth, min_panel_LTEs_uniform,label='uniform',width=boxwidth, zorder=2, edgecolor='k', color='r')
        # ax.hlines([30], xmin=-1, xmax=npan, color='k', alpha=0.5, zorder=1.1)
        # ax.set_yscale('log')
        # ax.set_ylim(1,10000)
        # # ax.set_ylim(1,55)
        # ax.xaxis.set_major_locator(tck.MultipleLocator()) # thanks stack exchange!
        # ax.set_xlabel('panel number', fontsize=fontsize)
        # ax.set_ylabel('lifetime (yrs)', fontsize=fontsize)
        # ax.set_xlim(-0.5,npan-0.25)
        # ax.legend(ncols=3, fontsize=fontsize-4)
        # fig.savefig(f'imgs/default_lifetime_barplot_Qdes{Qdes}',dpi=300)
        # # fig.show()
        # plt.close(fig)


        # # # LWT_therm_mod_helpers.plot_results(model_hvy, dTs_uniform.reshape(dTs_uniform.shape), label_name='temperature difference (C)', savename=f'dT_Qdes{Qdes}_uniform', vmin=None, vmax=None)
        # if Qdes==200:
        #     LWT_therm_mod_helpers.plot_results(model_hvy, LTEs_stage1pt5.reshape(dTs_uniform.shape), label_name='lifetimes (yrs)', savename=f'LTEs_Qdes{Qdes}_stage1pt5', vmin=0, vmax=100)
        #     LWT_therm_mod_helpers.plot_results(model_hvy, LTEs_stage2.reshape(dTs_uniform.shape), label_name='lifetimes (yrs)', savename=f'LTEs_Qdes{Qdes}_heuristic', vmin=0, vmax=100)

        ## check series of criteria and print out results
        params_stage_1pt5   =['Npanels', 'Qfluid', 'mflow', 'Qsinc', 'Ndisabled_heliostats','Nsub30']
        params_stage_2      =['Npanels', 'Qfluid_W', 'mflow', 'Qsinc_W', 'N_heliostats_used','Nsub30']

        #### report filestrings
        filestring_stage1pt5  =f'reports/heuristic_comparison_study/Qdes{Qdes}_npan{npan}_W18.0_H15.0_stage_1pt5_80yr_report.json'
        # filestring_stage1pt5  =f'heuristic_comparison_study/Qdes{Qdes}nom_npan{npan}_W18.0_H15.0_stage_1pt5_report.json'
        
        filestring_stage2     =f'reports/heuristic_comparison_study/compare_3D_heuristic_{Qdes}_results.json' # comment out to run 174 stage
        
        filestring_baseline  =f'reports/heuristic_comparison_study/Qdes{Qdes}_npan{npan}_W18.0_H15.0_uniform_report.json'
        # filestring_baseline  =f'heuristic_comparison_study/Qdes{Qdes}nom_npan{npan}_W18.0_H15.0_stage_uniform_report.json'
        ####

        N_total              =N_totals[Qdess.index(Qdes)]

        print('\n', 'uniform ----- stage 1.5 ----- stage 2 -----'   )
        for param_stage_1pt5, param_stage_2 in zip(params_stage_1pt5, params_stage_2):
            if param_stage_1pt5 == 'Nsub30':
                value_stage_uniform      =np.sum(min_panel_LTEs_uniform < 30)          
                value_stage_1pt5         =np.sum(min_panel_LTEs_stage1pt5 < 30)          
                value_stage_2            =np.sum(min_panel_LTEs_stage2 < 30)          
            else:
                value_stage_uniform      =helpers.json_to_np(filestring_baseline, param_stage_1pt5)
                value_stage_1pt5         =helpers.json_to_np(filestring_stage1pt5, param_stage_1pt5)
                value_stage_2            =helpers.json_to_np(filestring_stage2   , param_stage_2)

            if param_stage_2 =='mflow':
                value_stage_2 = np.sum(np.array(value_stage_2))
            elif param_stage_1pt5 == 'Ndisabled_heliostats':
                value_stage_1pt5 =N_total -value_stage_1pt5  
                value_stage_uniform =N_total - value_stage_uniform 

            print(f'{value_stage_uniform:10.0f} ----- {value_stage_1pt5:10.0f} ----- {value_stage_2:10.0f}')
       
        markersize =6
        # plot uniform stage
        ax2.scatter(Qdes*np.ones(min_panel_LTEs_uniform.size), min_panel_LTEs_uniform, s=markersize, color='r')
        # plot stage 1.5
        ax2.scatter(Qdes*np.ones(min_panel_LTEs_stage1pt5.size)+2, min_panel_LTEs_stage1pt5, s=markersize, color='k')
        # # plot stage 2
        ax2.scatter(Qdes*np.ones(min_panel_LTEs_stage2.size)+4, min_panel_LTEs_stage2, s=markersize, color='b')
        if Qdes == Qdess[-1]:
            # plot uniform stage
            ax2.scatter(Qdes*np.ones(min_panel_LTEs_uniform.size), min_panel_LTEs_uniform, s=markersize, label='uniform-SPT', color='r')
            # plot stage 1.5
            ax2.scatter(Qdes*np.ones(min_panel_LTEs_stage1pt5.size)+2, min_panel_LTEs_stage1pt5, s=markersize, label='IA-SPT', color='k')
            # # plot stage 2
            ax2.scatter(Qdes*np.ones(min_panel_LTEs_stage2.size)+4, min_panel_LTEs_stage2, s=markersize, label='IA-heuristic', color='b')

    # ax3.set_xlabel('receiver power to fluid [MWth]')

    ax2.set_xlabel('design power (MWth)', fontsize=fontsize)
    ax2.set_ylabel('lifetime (yrs)', fontsize=fontsize)
    ax2.axhline(y=30,linestyle='--',color='gray',linewidth = 1)

    # ax2.set_yscale('log')
    ax2.set_ylim(bottom=1, top=100)

    ## add the global minimum line
    ax2.plot(Qdess, uniform_rec_mins, color='r', linestyle='--', label='uniform-SPT, min')
    ax2.plot(np.array(Qdess)+2, stage1pt5_rec_mins, color='k', linestyle='--', label='IA-SPT, min')
    ax2.plot(np.array(Qdess)+4, stage2_rec_mins, color='b', linestyle='--', label='IA-heuristic, min')
    

    # set the legend order
    #get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    #specify order of items in legend
    order = [0,3,1,4,2,5]
    # order = [0,2,1,3] # if only plotting uniform and informed spt method

    # #add legend to plot
    # ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
    #            loc='lower left', ncols=3, columnspacing=0, handletextpad=0.02, borderpad=0.1)
    #add legend to plot
    ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
               loc='lower left', ncols=2, columnspacing=0, handletextpad=0.02, borderpad=0.1)

    # ax2.legend(loc='lower left',ncols=3,columnspacing=0,handletextpad=0.01,borderpad=0.1)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(fname=('imgs/method_comparison_per_design_power.png'),dpi=300)
    plt.show()
