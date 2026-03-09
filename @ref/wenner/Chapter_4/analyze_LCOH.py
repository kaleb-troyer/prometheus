"""
created by jwenner on 11/10/2025 to estimate LCOH for a given receiver design
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

    def parabolic_fun(x, p1, p2, p3):
        return p1*(x-p2)**2 + p3
    
    ## old interpolation that relied on DNI. Not very accurate because lower DNI shouldn't always equate to lower efficiency    
    # def interp_Qs(model, DNI_samples):
    #     '''
    #     Takes solved model and creates curve fits for receiver power and pump power vs dni.
    #     Makes season-specific curve fits for variables
    #     --- inputs:
    #     model       - steady state analysis model that has been solved for the desired day.
    #     DNI_samples - array of DNIs (W/m2) at which we want to predict Qfluid and Qpump. Need not be the same as the simulated DNI points
    #     --- returns:
    #     Qfluids_sample - an array of fluid powers (W) for each DNI sample point
    #     Qpumps_sample  - an array of total pump power required (W) for each DNI sample point
    #     '''
    #     ## put the values we want in new variables
    #     Qfluids_model=model.results.Qfluid
    #     Qpumps_model =model.results.Qpump
    #     DNIs_model   =model.results.dni
 

    #     ### Qfluid: interpolate fluid values at the DNI levels dictated by the season
    #     print('note: assuming DNI and Qfluid always increase/decrease simulataneously')
    #     ps, pcovs                   =curve_fit(parabolic_fun, DNIs_model, Qfluids_model, maxfev=100000) # high maxfev but necessary for the winter fit
    #     p1, p2, p3                  =ps
    #     Qfluids_samples                                 =parabolic_fun(DNI_samples, p1, p2, p3)
    #     Qfluids_samples[DNI_samples < model.dni_cutoff] =0

    #     ## plot the fit as a logic check
    #     fig,ax  =plt.subplots()
    #     fontsize=14
    #     ax.scatter(DNIs_model, Qfluids_model/1e6, label='model')
    #     ax.scatter(DNI_samples, Qfluids_samples/1e6, label='from fit')
    #     ax.plot(np.linspace(DNIs_model.min(),DNIs_model.max()), parabolic_fun(np.linspace(DNIs_model.min(),DNIs_model.max()),p1, p2, p3)/1e6, label='fit')

    #     ax.set_xlabel('Direct Normal Irradiance [W/m2]', fontsize=fontsize)
    #     ax.set_ylabel('power to fluid (MWth)', fontsize=fontsize)
    #     ax.tick_params('both', labelsize=fontsize)
    #     ax.legend(fontsize=fontsize)
    #     # fig.savefig('imgs/power_dni_fit_summer.png',dpi=300)
    #     plt.show()
    #     plt.close()


    #     ## interpolate pump values at the DNI levels dictated by the summer solstice
    #     pump_ps, pcovs                   =curve_fit(parabolic_fun, Qfluids_model, Qpumps_model)
    #     pump_p1, pump_p2, pump_p3        =pump_ps

    #     Qpumps_samples                      =parabolic_fun(Qfluids_samples, pump_p1, pump_p2, pump_p3)
    #     Qpumps_samples[Qfluids_samples < 1] =0
        

    #     ## check plot to ensure validity
    #     fig,ax  =plt.subplots()
    #     fontsize=14
    #     ax.scatter(Qfluids_model/1e6, Qpumps_model/1e6, label='model')
    #     ax.scatter(Qfluids_samples/1e6, Qpumps_samples/1e6, label='from fit')
    #     ax.plot(np.linspace(Qfluids_model.min(),Qfluids_model.max())/1e6, 
    #             parabolic_fun(np.linspace(Qfluids_model.min(),Qfluids_model.max()),pump_p1, pump_p2, pump_p3)/1e6, label=' model fit')

    #     ax.set_xlabel('power to fluid (MWth)', fontsize=fontsize)
    #     ax.set_ylabel('required pumping power (MWth)', fontsize=fontsize)
    #     ax.tick_params('both', labelsize=fontsize)
    #     ax.legend(fontsize=fontsize)
    #     # fig.savefig('imgs/pump_power_curve_summer.png', dpi=300)
    #     plt.show()
    #     plt.close()



        # return Qfluids_samples, Qpumps_samples
    

    def interp_Qs(model, sample_times, sample_DNIs):
        '''
        Takes solved model and creates curve fits for receiver power and pump power vs dni.
        Makes season-specific curve fits for variables
        --- inputs:
        model       - steady state analysis model that has been solved for the desired day.
        DNI_samples - array of DNIs (W/m2) at which we want to predict Qfluid and Qpump. Need not be the same as the simulated DNI points
        --- returns:
        Qfluids_sample - an array of fluid powers (W) for each DNI sample point
        Qpumps_sample  - an array of total pump power required (W) for each DNI sample point
        '''
        ## put the values we want in new variables
        Qfluids_model   =model.results.Qfluid
        Qpumps_model    =model.results.Qpump
        DNIs_model      =model.results.dni
        Qinc_model      =model.results.Qsinc   # incident power for each timestep
        eta_model       =model.results.eta  # receiver's thermal efficiency (Qfluid/Qsinc)


        ### Qfluid: interpolate field optical efficiency and receiver efficiency to equate
        # print('note: assuming DNI and Qfluid always increase/decrease simulataneously')
        # ps, pcovs                   =curve_fit(parabolic_fun, DNIs_model, Qfluids_model, maxfev=100000) # high maxfev but necessary for the winter fit
        # p1, p2, p3                  =ps
        # Qfluids_samples                                 =parabolic_fun(DNI_samples, p1, p2, p3)
        # Qfluids_samples[DNI_samples < model.dni_cutoff] =0
        eta_field_samples     =np.interp(sample_times, model.results.time, model.results.eta_field)

        ## calculate the sampled incident power
        n_hstats_used   =pd.read_csv(model.receiver.layout_file).index.max()+1-model.SF.Ndisable    # get total number of mirrors, subtract the number of disabled heliostats
        print('assuming heliostats each have area 144 m2')
        A_sf            =144.4*n_hstats_used
        Qsinc_samples   =A_sf*sample_DNIs.squeeze()*eta_field_samples

        ## interpolate the receiver efficiency
        ps, pcovs                   =curve_fit(parabolic_fun, Qinc_model, eta_model, maxfev=100000) # high maxfev but necessary for the winter fit
        p1, p2, p3                  =ps
        eta_samples                 =parabolic_fun(Qsinc_samples, p1, p2, p3)
        Qfluids_samples             =Qsinc_samples*eta_samples*model.receiver.solar_abs
        Qfluids_samples[sample_DNIs.squeeze() < model.dni_cutoff] =0

        ## plot the fit as a logic check
        fig,ax  =plt.subplots()
        fontsize=14
        ax.scatter(model.results.time, Qfluids_model/1e6, label='model')
        ax.scatter(sample_times, Qfluids_samples/1e6, label='from fit')
        # ax.plot(np.linspace(DNIs_model.min(),DNIs_model.max()), parabolic_fun(np.linspace(DNIs_model.min(),DNIs_model.max()),p1, p2, p3)/1e6, label='fit')

        ax.set_xlabel('time (hr)', fontsize=fontsize)
        ax.set_ylabel('power to fluid (MWth)', fontsize=fontsize)
        ax.tick_params('both', labelsize=fontsize)
        ax.legend(fontsize=fontsize)
        fig.savefig('imgs/default_save_fit_of_Qfluid.png',dpi=300)
        plt.show()
        plt.close()


        ## interpolate pump values at the DNI levels dictated by the summer solstice
        pump_ps, pcovs                   =curve_fit(parabolic_fun, Qfluids_model, Qpumps_model)
        pump_p1, pump_p2, pump_p3        =pump_ps

        Qpumps_samples                      =parabolic_fun(Qfluids_samples, pump_p1, pump_p2, pump_p3)
        Qpumps_samples[Qfluids_samples < 1] =0
        

        ## check plot to ensure validity
        fig,ax  =plt.subplots()
        fontsize=14
        ax.scatter(Qfluids_model/1e6, Qpumps_model/1e6, label='model')
        ax.scatter(Qfluids_samples/1e6, Qpumps_samples/1e6, label='from fit')
        ax.plot(np.linspace(Qfluids_model.min(),Qfluids_model.max())/1e6, 
                parabolic_fun(np.linspace(Qfluids_model.min(),Qfluids_model.max()),pump_p1, pump_p2, pump_p3)/1e6, label=' model fit')

        ax.set_xlabel('power to fluid (MWth)', fontsize=fontsize)
        ax.set_ylabel('required pumping power (MWth)', fontsize=fontsize)
        ax.tick_params('both', labelsize=fontsize)
        ax.legend(fontsize=fontsize)
        fig.savefig('imgs/default_save_pump_power_curve.png', dpi=300)
        plt.show()
        plt.close()



        return Qfluids_samples, Qpumps_samples
    
    ## initialize yearly arrays
    Qpumps  =[]
    Qfluids =[]
    days    =[]

    ## set up the full steady state analysis object
    model =steady_state_analysis_jwenn.SteadyStateAnalysis()

    input_name ='receivers/AR_study/billboard_200Qdes_0.83AR'
    model.receiver.load_from_json(input_name)

    # set the axial nodes to be equal to the number of axial nodes used in the optical simulation, which is best practice according to Figure 3.10 in dissertation
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

    ## ------------------ summer solstice ------------------ ##
    ## run for single timepoint
    AR          =model.receiver.H/model.receiver.D
    case_string =f'Qdes{model.receiver.Qdes}_W{model.receiver.D:.1f}_H{model.receiver.H:.2f}_AR{AR:.2f}_aspect_study_summer'
    # t_start     =timeit.default_timer()
    # model.solve(verbosity=1)
    # t_end       =timeit.default_timer()
    # t_solve     =t_end-t_start


    ## get the seasonal DNI reference
    with open('seasonal_DNI',) as f:
        DNI_seasons=json.load(f)

    ### get summer performance
    model.analysis_days                     =[172] # 172 can manually set this but analysis_mode will default set this as well
    model.solve(verbosity=1)
    DNIs_summer_day                         =np.array(DNI_seasons['summer'])
    t_samples                               =np.arange(0,24)+0.5
    Qfluids_summer_day, Qpumps_summer_day   =interp_Qs(model, t_samples, DNIs_summer_day)  # interpolate Qs from the simulated timepoints on day 172
    days_summer                             =31+30+31
    Qfluids_summer  =np.tile(Qfluids_summer_day.squeeze(), days_summer)   # assume every day looks like the average DNI day we interpolated the performance curves at
    Qpumps_summer   =np.tile(Qpumps_summer_day.squeeze(), days_summer)

    ## get fall performance
    model.analysis_days                     =[264] # can manually set this but analysis_mode will default set this as well
    model.solve(verbosity=1)
    DNIs_fall_day                           =np.array(DNI_seasons['fall'])
    Qfluids_fall_day, Qpumps_fall_day       =interp_Qs(model, t_samples, DNIs_fall_day)
    days_fall                               =31+30+31
    Qfluids_fall  =np.tile(Qfluids_fall_day.squeeze(), days_fall)
    Qpumps_fall   =np.tile(Qpumps_fall_day.squeeze(), days_fall)

    ## get spring performance. No need to resimulate because we assume equinox
    DNIs_spring_day                         =np.array(DNI_seasons['spring'])
    Qfluids_spring_day, Qpumps_spring_day   =interp_Qs(model, t_samples, DNIs_spring_day)
    days_spring                             =28+31+30
    Qfluids_spring                          =np.tile(Qfluids_spring_day.squeeze(), days_spring)
    Qpumps_spring                           =np.tile(Qpumps_spring_day.squeeze(), days_spring)

    ## get winter performance
    model.analysis_days                     =[355] # can manually set this but analysis_mode will default set this as well
    model.solve(verbosity=1)
    DNIs_winter_day                         =np.array(DNI_seasons['winter'])
    Qfluids_winter_day, Qpumps_winter_day   =interp_Qs(model, t_samples, DNIs_winter_day)
    days_winter                             =31+30+31
    Qfluids_winter                          =np.tile(Qfluids_winter_day.squeeze(), days_winter)
    Qpumps_winter                           =np.tile(Qpumps_winter_day.squeeze(), days_winter)


    ## combine all seasons power to HTF and required pump power
    Qfluids_year =np.concatenate((Qfluids_spring, Qfluids_summer, Qfluids_fall, Qfluids_winter)) # this is a value in watts
    Qpumps_year  =np.concatenate((Qpumps_spring, Qpumps_summer, Qpumps_fall, Qpumps_winter))

    ## ---------- estimate LCOH with all information -------------------- ## 
    times=np.ones(8760)*3600 # one timestep for every hour in a year, make the timestep in seconds
    rec_cost_model=tower_model.costModelFR(Qdes=200e6, Htow=180, Hrec=model.receiver.H, Wrec=model.receiver.D, Wpanel=model.receiver.D/model.receiver.Npanels, D_o=50.8e-3, th=1.25e-3, material='A230', A_Hstats=424e3, 
                         A_land=800e3, N_life=30, N_repl=0, Qdot_HTFs=Qfluids_year.flatten(), times=times, P_el_pumps=Qpumps_year.flatten(), eta_PBII=0.45)
    LCOH          =rec_cost_model.calc_LCOH()*1e6*3600 # convert from euros/W-s to euros/MWh
    print(f'the calculated LCOH is {LCOH} euros/MWh')
    print('--')
    print(f'annual energy is {(Qfluids_year.sum()/1e9):.2f} GWh')
    print('--')


