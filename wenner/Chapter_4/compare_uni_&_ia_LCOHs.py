"""
created by jwenner on 12.8.25 to plot uniform and informed aiming scheme LCOH results for a given design power
"""
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
from matplotlib.ticker import LogLocator, FormatStrFormatter, LogFormatter, FixedLocator, MaxNLocator
from scipy.optimize import curve_fit
import tower_model


if __name__ == '__main__':
    fig,ax =plt.subplots(tight_layout=True)
    ARs             =[0.23, 0.26, 0.30, 0.34, 0.40, 0.47, 0.56, 0.68, 0.83, 1.05, 1.38, 1.88, 2.70]

    CAPEXs_UNI          =[]
    OPEXs_UNI           =[]
    Replacements_UNI    =[] # replacements are normalized by the total number of panels in the receiver
    Q_yearly_nets_UNI   =[]
    eta_f_des_UNI       =[]
    P_pumps_total_UNI   =[]
    nstats_used_UNI     =[]
    eta_summer_avg_UNI  =[]
    eta_winter_avg_UNI  =[]
    eta_fall_avg_UNI    =[]
    LCOH_UNIs           =[]

    CAPEXs_IA          =[]
    OPEXs_IA           =[]
    Replacements_IA    =[]
    Q_yearly_nets_IA   =[]
    eta_f_des_IA       =[]
    P_pumps_total_IA   =[]
    nstats_used_IA     =[]
    eta_summer_avg_IA  =[]
    eta_winter_avg_IA  =[]
    eta_fall_avg_IA    =[]
    LCOH_IAs           =[]

    
    fontsize=16
    
    for AR in ARs:
        model           =steady_state_analysis_jwenn.SteadyStateAnalysis()
        Qdes            =220
        input_name      =f'receivers/AR_study/billboard_{Qdes}Qdes_{AR:.2f}AR'
        model.receiver.load_from_json(input_name)
        # case_string_UNI  =f'Qdes{model.receiver.Qdes}_W{model.receiver.D:.1f}_H{model.receiver.H:.2f}_AR{AR:.2f}_LCOH_results_uniform'
        case_string_UNI  =f'Qdes{model.receiver.Qdes}_W{model.receiver.D:.1f}_H{model.receiver.H:.2f}_AR{AR:.2f}_LCOH_results_uniform'
        case_string_IA   =f'Qdes{model.receiver.Qdes}_W{model.receiver.D:.1f}_H{model.receiver.H:.2f}_AR{AR:.2f}_LCOH_results'

        ## get all inputs from json input file for uniform
        inputName = input_name + '.json'
        outputName= f'reports/{case_string_UNI}.json'
        with open(outputName,) as f:
            report_dict_UNI=json.load(f)
        


        ## get all inputs from json input file for uniform
        inputName = input_name + '.json'
        outputName= f'reports/{case_string_IA}.json'
        with open(outputName,) as f:
            report_dict_IA=json.load(f)
        
    
        
        ## resolve the cost model and get some breakdown elements for UNI results
        rec_cost_model_UNI  =tower_model.costModelFR(Qdes=report_dict_UNI['Qdes']*1e6, Htow=report_dict_UNI['Htower'], Hrec=report_dict_UNI['H'], Wrec=report_dict_UNI['D'],
                                                Wpanel=report_dict_UNI['D']/report_dict_UNI['Npanels'], D_o=report_dict_UNI['tube_OD'], th=report_dict_UNI['tube_twall'], material='A230', A_Hstats=report_dict_UNI['n_hstats_used']*144, 
                                                A_land=report_dict_UNI['n_hstats_used']*144*7.54, N_life=30, N_repl=report_dict_UNI['N_replaced'], Qdot_HTFs=np.array(report_dict_UNI['Qfluids_yearly']), times=np.ones(8760)*3600,
                                                P_el_pumps=np.array(report_dict_UNI['Qpumps_yearly'])*.412, eta_PBII=0.45)
        LCOH_UNI                   =rec_cost_model_UNI.calc_LCOH()*1e6*3600 # i needed to annualize the operating expenses
        ax.scatter(AR, LCOH_UNI, color='red', edgecolors='k', marker='D') if AR != 2.70 else ax.scatter(AR, LCOH_UNI, color='red', edgecolors='k', marker='D', label='uniform')
        
        ## resolve the cost model and get some breakdown elements for IA results
        rec_cost_model_IA   =tower_model.costModelFR(Qdes=report_dict_IA['Qdes']*1e6, Htow=report_dict_IA['Htower'], Hrec=report_dict_IA['H'], Wrec=report_dict_IA['D'],
                                                Wpanel=report_dict_IA['D']/report_dict_IA['Npanels'], D_o=report_dict_IA['tube_OD'], th=report_dict_IA['tube_twall'], material='A230', A_Hstats=report_dict_IA['n_hstats_used']*144, 
                                                A_land=report_dict_IA['n_hstats_used']*144*7.54, N_life=30, N_repl=report_dict_IA['N_replaced'], Qdot_HTFs=np.array(report_dict_IA['Qfluids_yearly']), times=np.ones(8760)*3600,
                                                P_el_pumps=np.array(report_dict_IA['Qpumps_yearly'])*.412, eta_PBII=0.45)
        LCOH_IA                   =rec_cost_model_IA.calc_LCOH()*1e6*3600 # i needed to annualize the operating expenses
        ax.scatter(AR, LCOH_IA, color='b', edgecolors='k', marker='D') if AR != 2.70 else ax.scatter(AR, LCOH_IA, color='b', edgecolors='k', marker='D', label='informed')

        ## plot a TEC breakdown if desired
        # if AR==0.83:
        #     TEC_dict_UNI               ={}
        #     TEC_dict_UNI['solar field']=[rec_cost_model_UNI.C_SF/1e6]
        #     TEC_dict_UNI['tower']      =[rec_cost_model_UNI.C_tow/1e6]
        #     TEC_dict_UNI['receiver']   =[rec_cost_model_UNI.C_rec/1e6]
        #     TEC_dict_UNI['piping']     =[rec_cost_model_UNI.C_pip/1e6]
        #     TEC_dict_UNI['pump']       =[rec_cost_model_UNI.C_pump/1e6]
        #     TEC_dict_UNI['controls']   =[rec_cost_model_UNI.C_contr/1e6]
        #     TEC_dict_UNI['spares']     =[rec_cost_model_UNI.C_spare/1e6]
        #     TEC_df_UNI                 =pd.DataFrame(TEC_dict_UNI)

        #     fig2,ax2=plt.subplots()
        #     fontsize=14
        #     ax2.bar(TEC_df_UNI.columns, TEC_df_UNI.iloc[0])
        #     ax2.set_xlabel('TEC subcategory',fontsize=fontsize)
        #     ax2.set_ylabel('cost (million euros)', fontsize=fontsize)
        #     ax2.tick_params(labelsize=fontsize-4)
        #     # fig2.savefig(f'imgs/lifetime_analysis_for_{AR:.2f}AR.png', dpi=300)
        #     # plt.show()
        #     plt.close()
        # #
        #     ## plot efficiency throughout the day for both aiming styles
        #     fig,ax=plt.subplots()
        #     time_normd  =np.linspace(0,1,len(report_dict_IA['summer_opt_eff']))
        #     ax.scatter(time_normd, report_dict_UNI['summer_opt_eff'], edgecolors='k', marker='D' ,color='r', label='uniform', s=20)
        #     ax.plot(time_normd, report_dict_UNI['summer_opt_eff'], color='r', linestyle='--', linewidth=1.5)
        #     ax.scatter(time_normd, report_dict_IA['summer_opt_eff'], edgecolors='k', marker='D' ,color='b', label='informed', s=20)
        #     ax.plot(time_normd, report_dict_IA['summer_opt_eff'], color='b', linestyle='--', linewidth=1.5)
        #     ax.set_xlabel('hour / total hours',fontsize=fontsize)
        #     ax.set_ylabel('field efficiency', fontsize=fontsize)
        #     ax.legend(fontsize=fontsize)
        #     ax.tick_params(labelsize=fontsize-2)
        #     ax.grid(True, which='both',alpha=0.3)
        #     plt.savefig('imgs/eta_profile_summer_ia_vs_uni.png', dpi=300)
        #     plt.show()
        #     plt.close()
        #     ##

        ## append to various tracker lists for later use
        CAPEXs_UNI.append(rec_cost_model_UNI.CAPEX/1e6)
        OPEXs_UNI.append(rec_cost_model_UNI.N_life*rec_cost_model_UNI.OPEX/1e6)
        Replacements_UNI.append(report_dict_UNI['N_replaced']/model.receiver.Npanels)
        Q_yearly_nets_UNI.append(rec_cost_model_UNI.QtoHTF/1e9/3600 )
        eta_f_des_UNI.append(np.array(report_dict_UNI['summer_opt_eff']).max())
        P_pumps_total_UNI.append(rec_cost_model_UNI.P_el_pumps.sum()/1e9)
        nstats_used_UNI.append(report_dict_UNI['n_hstats_used'])
        eta_summer_avg_UNI.append(np.array(report_dict_UNI['summer_opt_eff']).mean())
        eta_winter_avg_UNI.append(np.array(report_dict_UNI['winter_opt_eff']).mean())
        eta_fall_avg_UNI.append(np.array(report_dict_UNI['fall_opt_eff']).mean())
        LCOH_UNIs.append(LCOH_UNI)

        CAPEXs_IA.append(rec_cost_model_IA.CAPEX/1e6)
        OPEXs_IA.append(rec_cost_model_IA.N_life*rec_cost_model_IA.OPEX/1e6)
        Replacements_IA.append(report_dict_IA['N_replaced']/model.receiver.Npanels)
        Q_yearly_nets_IA.append(rec_cost_model_IA.QtoHTF/1e9/3600 )
        eta_f_des_IA.append(np.array(report_dict_IA['summer_opt_eff']).max())
        P_pumps_total_IA.append(rec_cost_model_IA.P_el_pumps.sum()/1e9)
        nstats_used_IA.append(report_dict_IA['n_hstats_used'])
        eta_summer_avg_IA.append(np.array(report_dict_IA['summer_opt_eff']).mean())
        eta_winter_avg_IA.append(np.array(report_dict_IA['winter_opt_eff']).mean())
        eta_fall_avg_IA.append(np.array(report_dict_IA['fall_opt_eff']).mean())
        LCOH_IAs.append(LCOH_IA)


    print(f'LCOH min. for IA aiming is: {np.array(LCOH_IAs).min():.2f} at AR: {ARs[np.argmin(np.array(LCOH_IAs))]}')
    print(f'LCOH min. for UNI aiming is: {np.array(LCOH_UNIs).min():.2f} at AR: {ARs[np.argmin(np.array(LCOH_UNIs))]}')
    print('\n')
    print(f'CAPEX min. for IA aiming is: {np.array(CAPEXs_IA).min():.2f} at AR: {ARs[np.argmin(np.array(CAPEXs_IA))]}')
    print(f'CAPEX min. for UNI aiming is: {np.array(CAPEXs_UNI).min():.2f} at AR: {ARs[np.argmin(np.array(CAPEXs_UNI))]}')
    print('\n')
    print(f'OPEX min. for IA aiming is: {np.array(OPEXs_IA).min():.2f} at AR: {ARs[np.argmin(np.array(OPEXs_IA))]}')
    print(f'OPEX min. for UNI aiming is: {np.array(OPEXs_UNI).min():.2f} at AR: {ARs[np.argmin(np.array(OPEXs_UNI))]}')
    print('\n')
    print(f'the average difference in design point efficiency is:{(np.array(eta_f_des_IA)-np.array(eta_f_des_UNI)).mean()*100:.2f}%')
    print(f'the average difference in yearly net energy is:{(np.array(Q_yearly_nets_IA)-np.array(Q_yearly_nets_UNI)).mean():.2f}')
    print(f'the average difference in OPEX is:{(np.array(OPEXs_IA)-np.array(OPEXs_UNI)).mean():.2f}')
    print(f'the average difference in LCOH is:{(np.array(LCOH_IAs)-np.array(LCOH_UNIs)).mean():.2f}')

    ax.set_xlabel('aspect ratio (H/W)', fontsize=fontsize)
    ax.set_ylabel('LCOH (euros/MWh)', fontsize=fontsize)
    ax.set_xscale('log')
    ax.set_ylim(bottom=25, top=31)
    custom_xticks = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xticks)
    ax.tick_params(labelsize=fontsize)
    ax.grid(True, which='both',alpha=0.3)
    ax.legend(fontsize=fontsize)
    plt.savefig(f'imgs/LCOH_vs_AR_uni_&_ia_{Qdes}', dpi=300)
    plt.show()
    plt.close()


    # plot the CAPEX and OPEX for different ARs - UNI
    ymax_capex=np.array(CAPEXs_UNI).max() + 5
    ymin_capex=np.array(CAPEXs_IA).min() - 5
    ymax_opex =np.array(OPEXs_UNI).max() + 5
    ymin_opex   =np.array(OPEXs_IA).min() - 5
    fontsize=16
    fig,ax=plt.subplots()
    lb1 =ax.scatter(ARs, CAPEXs_UNI, label='CAPEX', color='r', edgecolors='k', marker='D')
    ax.set_xscale('log')
    # ax2=ax.twinx()
    ax.set_ylim(bottom=ymin_opex, top=ymax_capex)
    # ax2.set_ylim(bottom=ymin_opex, top=ymax_opex)
    lb2 =ax.scatter(ARs, OPEXs_UNI, label='OPEX', color='k', edgecolors='k', marker='D')
    ax.set_xlabel('aspect ratio (H/W)', fontsize=fontsize)
    ax.set_ylabel('cost  (M€)', fontsize=fontsize)
    # ax2.set_ylabel('OPEX (M€)', fontsize=fontsize)
    # log crap
    custom_xticks = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xticks)
    #
    ax.grid(True, which='both',alpha=0.3)
    ax.tick_params(labelsize=fontsize-2)
    # ax2.tick_params(labelsize=fontsize-2)
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(fontsize=fontsize)
    # ax2.legend(lines + lines2, labels + labels2, loc=0, fontsize=fontsize-2)    
    plt.savefig(f'imgs/EXs_vs_ARs_uni_{Qdes}.png', dpi=300)
    plt.show()
    plt.close()

    # plot the CAPEX and OPEX for different ARs - IA
    fig,ax=plt.subplots(tight_layout=True)
    lb1 =ax.scatter(ARs, CAPEXs_IA, label='CAPEX', color='r', edgecolors='k', marker='D')
    ax.set_xscale('log')
    # ax.set_ylim(bottom=ymin_opex, top=ymax_capex)
    # ax.set_ylim(bottom=ymin_opex, top=ymax_capex)
    # ax.set_ylim(bottom=ymin_opex, top=ymax_capex) # most recent
    ax2=ax.twinx()
    ax.set_ylim(bottom=150, top=235)
    ax2.set_ylim(bottom=60, top=105)
    lb2 =ax2.scatter(ARs, OPEXs_IA, label='OPEX', color='k', edgecolors='k', marker='D')
    ax.set_xlabel('aspect ratio (H/W)', fontsize=fontsize)
    ax.set_ylabel('CAPEX (M€)', fontsize=fontsize)
    ax2.set_ylabel('OPEX (M€)', fontsize=fontsize)
    ## vertical line splacement
    ax.axvline(ARs[np.argmin(np.array(OPEXs_IA))], linestyle='--', color='k',alpha=0.7)
    ax.axvline(ARs[np.argmin(np.array(CAPEXs_IA))], linestyle='--', color='r',alpha=0.7)
    ##
    # ax.legend(fontsize=fontsize)
    # #ax2.legend(fontsize=fontsize-4)
    # log crap
    custom_xticks = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xticks)
    #
    ax.grid(True, which='both',alpha=0.3)
    ax.tick_params(labelsize=fontsize-2)
    ax2.tick_params(labelsize=fontsize-2)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='right', fontsize=fontsize-2)    
    plt.savefig(f'imgs/EXs_vs_ARs_ia_{Qdes}.png', dpi=300)
    plt.show()
    plt.close()

    ## plot the number of replacements versus AR
    fig,ax=plt.subplots()
    ax.scatter(ARs, Replacements_UNI, edgecolors='k', marker='D', color='r', label='uniform')
    ax.scatter(ARs, Replacements_IA, edgecolors='k', marker='D', color='b', label='informed')
    ax.set_xlabel('aspect ratio (H/W)',fontsize=fontsize)
    ax.set_ylabel('panel replacements / receiver panels', fontsize=fontsize)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # get the log crap right
    ax.set_xscale('log')
    custom_xticks = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xticks)
    #
    custom_yticks =custom_xticks
    ax.set_yticks(custom_yticks)
    ax.set_yticklabels(custom_yticks)
    #
    ax.tick_params(labelsize=fontsize-2)
    ax.legend(fontsize=fontsize)
    ax.grid(True, which='both',alpha=0.3)
    ax.set_ylim(bottom=-0.05, top=1)
    plt.savefig(f'imgs/panel_replacements_uni_vs_ia_{Qdes}.png', dpi=300)
    plt.show()
    plt.close()

    ## plot the Qhtf versus AR
    fig,ax=plt.subplots()
    ax.scatter(ARs, Q_yearly_nets_UNI, edgecolors='k', marker='D', color='r', label='uniform')
    ax.scatter(ARs, Q_yearly_nets_IA, edgecolors='k', marker='D', color='b', label='informed')
    ax.set_xlabel('aspect ratio (H/W)',fontsize=fontsize)
    ax.set_ylabel('yearly net energy (GWh/year) ', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-2)
    ax.set_xscale('log')
    # log crap
    custom_xticks = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xticks)
    #
    ax.grid(True, which='both',alpha=0.3)
    plt.savefig(f'imgs/Q_htfs_uni_vs_ia_{Qdes}.png', dpi=300)
    plt.show()
    plt.close()

    ## plot the design efficiency for both aiming styles
    fig,ax=plt.subplots(tight_layout=True)
    ax.scatter(ARs, eta_f_des_UNI, edgecolors='k', marker='D' ,color='r', label='uniform')
    ax.scatter(ARs, eta_f_des_IA, edgecolors='k', marker='D' ,color='b', label='informed')
    ax.set_xlabel('aspect ratio (H/W)',fontsize=fontsize)
    ax.set_ylabel('design point efficiency', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-2)
    ax.set_xscale('log')
    custom_xticks = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xticks)
    ax.tick_params(axis='x', labelrotation=0)
    ax.set_xlim(left=0.2, right=3.2)
    ax.set_ylim(bottom=0.42, top=0.6)
    ax.grid(True, which='both',alpha=0.3)
    plt.savefig(f'imgs/eta_des_uni_vs_ia_{Qdes}.png', dpi=300)
    plt.show()
    plt.close()

    ## plot the number of heliostats for both aiming styles
    fig,ax=plt.subplots(tight_layout=True)
    ax.scatter(ARs, nstats_used_UNI, edgecolors='k', marker='D' ,color='r', label='uniform')
    ax.scatter(ARs, nstats_used_IA, edgecolors='k', marker='D' ,color='b', label='informed')
    ax.set_xlabel('aspect ratio (H/W)',fontsize=fontsize)
    ax.set_ylabel('number of heliostats used', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-2)
    ax.set_xscale('log')
    custom_xticks = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xticks)
    ax.tick_params(axis='x', labelrotation=0)
    ax.set_xlim(left=0.2, right=3.2)
    ax.grid(True, which='both',alpha=0.3)
    plt.savefig(f'imgs/nhstats_uni_vs_ia_{Qdes}.png', dpi=300)
    plt.show()
    plt.close()

    ## plot the seasonal average efficiencies for both aiming styles
    fig,ax=plt.subplots(tight_layout=True)
    ax.plot(ARs, eta_summer_avg_UNI, marker='D' ,color='r', label='uniform,summer',linestyle='--')
    ax.plot(ARs, eta_summer_avg_IA, marker='D' ,color='b', label='informed,summer',linestyle='--')
    ax.set_xlabel('aspect ratio (H/W)',fontsize=fontsize)
    ax.set_ylabel('average efficiency', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-2)
    ax.set_xscale('log')
    custom_xticks = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xticks)
    ax.tick_params(axis='x', labelrotation=0)
    ax.set_xlim(left=0.2, right=3.2)
    ax.grid(True, which='both',alpha=0.3)
    plt.savefig(f'imgs/summer_eta_means_uni_vs_ia_{Qdes}.png', dpi=300)
    plt.show()
    plt.close()



