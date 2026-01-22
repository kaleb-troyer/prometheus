"""
created by jwenner on 11.18.25 to plot aspect ratio study results
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
import matplotlib.ticker as tck
from scipy.optimize import curve_fit
import tower_model

if __name__ == '__main__':
    fig,ax =plt.subplots(tight_layout=True)
    ARs             =[0.23, 0.26, 0.30, 0.34, 0.40, 0.47, 0.56, 0.68, 0.83, 1.05, 1.38, 1.88, 2.70]
    CAPEXs          =[]
    OPEXs           =[]
    Replacements    =[]
    Q_yearly_nets   =[]
    
    fontsize=14
    
    for AR in ARs:
        model =steady_state_analysis_jwenn.SteadyStateAnalysis()
        input_name =f'receivers/AR_study/billboard_200Qdes_{AR:.2f}AR'
        model.receiver.load_from_json(input_name)
        case_string =f'Qdes{model.receiver.Qdes}_W{model.receiver.D:.1f}_H{model.receiver.H:.2f}_AR{AR:.2f}_LCOH_results_uniform'

        ## get all inputs from json input file
        inputName = input_name + '.json'
        outputName= f'reports/{case_string}.json'

        with open(outputName,) as f:
            report_dict=json.load(f)
        
    
        
        ## resolve the cost model and get some breakdown elements
        rec_cost_model=tower_model.costModelFR(Qdes=report_dict['Qdes']*1e6, Htow=report_dict['Htower'], Hrec=report_dict['H'], Wrec=report_dict['D'],
                                                Wpanel=report_dict['D']/report_dict['Npanels'], D_o=report_dict['tube_OD'], th=report_dict['tube_twall'], material='A230', A_Hstats=report_dict['n_hstats_used']*144, 
                                                A_land=report_dict['n_hstats_used']*144*7.54, N_life=30, N_repl=report_dict['N_replaced'], Qdot_HTFs=np.array(report_dict['Qfluids_yearly']), times=np.ones(8760)*3600,
                                                P_el_pumps=np.array(report_dict['Qpumps_yearly'])*.412, eta_PBII=0.45)
        LCOH                   =rec_cost_model.calc_LCOH()*1e6*3600 # i needed to annualize the operating expenses
        # LCOH                    =report_dict['LCOH']
        ax.scatter(AR, LCOH, color='red', edgecolors='k', marker='D')

        if AR==0.83:
            TEC_dict               ={}
            TEC_dict['solar field']=[rec_cost_model.C_SF/1e6]
            TEC_dict['tower']      =[rec_cost_model.C_tow/1e6]
            TEC_dict['receiver']   =[rec_cost_model.C_rec/1e6]
            TEC_dict['piping']     =[rec_cost_model.C_pip/1e6]
            TEC_dict['pump']       =[rec_cost_model.C_pump/1e6]
            TEC_dict['controls']   =[rec_cost_model.C_contr/1e6]
            TEC_dict['spares']     =[rec_cost_model.C_spare/1e6]
            TEC_df                 =pd.DataFrame(TEC_dict)

            fig2,ax2=plt.subplots()
            fontsize=14
            ax2.bar(TEC_df.columns, TEC_df.iloc[0])
            ax2.set_xlabel('TEC subcategory',fontsize=fontsize)
            ax2.set_ylabel('cost (million euros)', fontsize=fontsize)
            ax2.tick_params(labelsize=fontsize-4)
            # fig2.savefig(f'imgs/lifetime_analysis_for_{AR:.2f}AR.png', dpi=300)
            # plt.show()
            plt.close()
        CAPEXs.append(rec_cost_model.CAPEX/1e6)
        OPEXs.append(rec_cost_model.N_life*rec_cost_model.OPEX/1e6)
        Replacements.append(report_dict['N_replaced'])
        Q_yearly_nets.append(rec_cost_model.Qdot_HTFs.sum()/1e9 )
    ax.set_xlabel('log(aspect ratio) (H/W)', fontsize=fontsize)
    ax.set_ylabel('LCOH (euros/MWh)', fontsize=fontsize)
    ax.set_xscale('log')
    ax.tick_params(which='both', labelbottom=True, labelsize=fontsize)
    ax.grid(True, which='both',alpha=0.3)
    # plt.savefig('imgs/LCOH_vs_AR', dpi=300)
    plt.show()
    plt.close()


    # plot the CAPEX and OPEX for different ARs
    fontsize=14
    fig,ax=plt.subplots()
    lb1 =ax.scatter(ARs, CAPEXs, label='CAPEX', color='r', edgecolors='k', marker='D')
    ax2=ax.twinx()
    lb2 =ax2.scatter(ARs, OPEXs, label='OPEX', color='k', edgecolors='k', marker='D')
    ax.set_xlabel('aspect ratio (H/W)', fontsize=fontsize)
    ax.set_ylabel('CAPEX  (M€)', fontsize=fontsize)
    ax2.set_ylabel('OPEX (M€)', fontsize=fontsize)
    # ax.legend(fontsize=fontsize-4)
    # ax2.legend(fontsize=fontsize-4)
    ax.tick_params(labelsize=fontsize-2)
    ax2.tick_params(labelsize=fontsize-2)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0, fontsize=fontsize-2)    

    # plt.savefig('imgs/EXs_vs_ARs.png', dpi=300)
    plt.show()
    plt.close()

    ## plot the number of replacements versus AR
    fig,ax=plt.subplots()
    ax.scatter(ARs, Replacements, edgecolors='k', marker='D', color='g')
    ax.set_xlabel('aspect ratio (H/W)',fontsize=fontsize)
    ax.set_ylabel('number of panel replacements', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-2)
    # plt.savefig('imgs/panel_replacements.png', dpi=300)
    plt.show()
    plt.close()

    ## plot the Qhtf versus AR
    fig,ax=plt.subplots()
    ax.scatter(ARs, Q_yearly_nets, edgecolors='k', marker='D', color='r')
    ax.set_xlabel('aspect ratio (H/W)',fontsize=fontsize)
    ax.set_ylabel('yearly net energy (GWh/year) ', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-2)
    # plt.savefig('imgs/Q_htfs.png', dpi=300)
    plt.show()
    plt.close()

