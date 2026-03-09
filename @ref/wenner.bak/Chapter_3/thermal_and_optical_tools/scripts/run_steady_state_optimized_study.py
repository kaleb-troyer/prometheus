
'''
jwenn version to help with implementation of new solarpilot method
'''
import sys
sys.path.append('..')
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


# plt.rcParams["figure.figsize"] = (18, 9)
# plt.rcParams.update({'font.size': 30})
# plt.rcParams['lines.markersize'] ** 2
# markersize=406

model = steady_state_analysis_jwenn.SteadyStateAnalysis()


#--- Read existing receiver design from file and/or update receiver design input parameters
inputName = 'cost_model_debug'
inputString = '../receivers/'+ inputName
model.receiver.load_from_json(inputString)

#--- Update numerical solution options as desired (all inputs have default values defined in the CylindricalReceiver class)

model.receiver.disc = settings.Discretization(5, 79, 50)   # number of r, theta, z nodes 
model.receiver.options.wall_detail = '2D' 
model.receiver.options.calculate_stress = True   # Calculate elastic thermal stress distributions?
model.receiver.flow_control_mode = 0             # 0 = Control each path independently
model.receiver.ntubesim = 3                      # Number of tubes simulated per panel


#--- Set up flux distribution options
model.check_for_existing_flux_distributions = False  # Check if flux distributions for this configuration have already been saved
model.save_flux_distributions = True
model.SF.saved_flux_direc = './flux_sims/'  # Directory to store flux distributions

#--- Set up model time resolution, DNI, and ambient conditions
model.analysis_mode = 'selected_days' #'design_day'  ,- was using this 1/27/25, og comment -># 'design_point', 'design_day', 'three_day' (summer solstice, equinox, winter solstice), 'selected_days', 'user_defined' # was using "selected_days"
model.analysis_days = [172] # can manually set this but analysis_mode will default set this as well
substeps=1                    # number of model substeps per hour
model.delta_hour = 1/substeps                # Time spacing between simulated time points (hr)
model.is_half_day = False           # Simulate only half of the day?
model.dni_setting = 'user-defined-noon'  
model.user_defined_dni = {172:950, 264:980, 355:930} # User-defined DNI at solar noon on each simulated day.  Used in conjunction with clearsky-DNI to set DNI per time point
model.ambient_setting = 'user-defined'
model.user_defined_Tamb = 25+273.15    # Constant user-defined ambient T (K)
model.user_defined_RH = 25             # Constant user-defined relative humidity
model.user_defined_vwind10 = 0         # Constant user-defined wind speed (m/s at 10m height)



#==========================================================================================================
if __name__ == "__main__":
    AR =model.receiver.H/model.receiver.D   
    case_string=f'Qdes{model.receiver.Qdes}_AR{AR:.2f}_W{model.receiver.D:.1f}_H{model.receiver.H:.2f}_aspect_study'

    #--- Solve and create plots
    start = timeit.default_timer()
    model.solve(verbosity = 1)
    print('Total solution time = %.2fs'%(timeit.default_timer()-start))
    daySel=172
    # Plot fluid T, peak wall T, absorbed flux, elastic stress vs. flow path location for a given time point
    imgName = "./imgs/"+ case_string + '_plot_profiles_in_path'
    print(imgName)
    steady_state_plots.plot_profiles_in_path(model, day = daySel, hour_offset = 0, use_all_tubes = False, savename = None )
    
    # Plot max fluid T, max wall T, max absorbed flux, and max equivalent elastic stress per panel vs. time points in one day
    imgName = "./imgs/"+ "_panelsVStime_"+str(substeps)+"substeps"
    steady_state_plots.plot_panels_vs_time_jwenn(model, day = daySel, savename = None)

    # Plot tube temperature profiles in specified panels (note this will use the last time point in the analysis... full temperature profiles per time point aren't currently saved)
    # i'm cheating. I create a cloned model, then solve it at design point. The last tube results can then be used to generate the plot I want
    d = model.find_design_point()

    ## make a results json
    outputString='./reports/'+case_string+'_report'
    helpers_thermal_model.make_results_file(inputString,outputString,model,d)
    ## make a time series json
    helpers_thermal_model.make_timeseries_results_file(inputString,outputString,model)

    # ## save the most recent dTs/Tfs using alternate data processing strategy. Not very useful if solving a multipoint timeseries
    # dTs_results, Tfs_results, qabs_results, Rs_results  =helpers_thermal_model.get_thermal_results(model.receiver)
    # thermal_results={}
    # thermal_results['dTs']=dTs_results.tolist()
    # thermal_results['Tfs']=Tfs_results.tolist()
    # with open(f'{outputString}_thermal_results.json', "w") as f:
    #     json.dump(thermal_results, f)
    
    ## plot a tube wall vs theta profile for select time -- requires solving the model again -- time consuming so commented out for now
    days, times, times_offset = model.find_allowable_time_points()
    # branchModel=model
    # branchModel.solve_time_point(doy=daySel, hour_of_day=times[d], verbosity=0) 
    # TambRad=getattr(branchModel.results, 'Tambrad')
    # imgName = "./imgs/"+ "_TvsTheta_"+str(substeps)+"substeps"
    # # steady_state_plots.plot_tube_wall_vs_theta(branchModel.receiver,panels = [7], axial_loc = 0.72, use_all_tubes = False, savename = imgName)    
    # steady_state_plots.plot_tube_wall_vs_theta(branchModel.receiver,panels = [0], axial_loc = 0.72, use_all_tubes = False)    


    # ntubes_panel=np.array(len(branchModel.receiver.tubes[0]))
    # nz=np.array(branchModel.receiver.disc.nz)
    # thetaPts=np.array(branchModel.receiver.disc.thetapts)
    # Tw = getattr(branchModel.receiver.tubes[7][0], "Tw")-273.15 # should yield a z x theta x radial array
    # sigmas = getattr(branchModel.receiver.tubes[7][0], 'stress_equiv')
    # names = ["ntubes_panel","nz","thetaPts","Tw", "sigmas"]
    # array_list = [ntubes_panel,nz,thetaPts,Tw, sigmas]
    # # helpers.np_to_json(names, array_list, "design_point_panel7_tube0_Tw_results.json")
    
    plt.show()
    hour_offset=0
    key='Tf'
    saveName='./imgs/'+'heatmap_hrOffset'+str(hour_offset)+'_key_'+str(key)
    steady_state_plots.plot_receiver_heatmap(model=model, key=key, day=days[0], hour_offset = hour_offset, plot_all_tubes = True, title = None, ylims = None, savename = None)
    plt.show()

    combineStr='max'       #specify if you want mean tube/panel temps plotted or max. use 'max' or <anything else>
    imgName="./imgs/"+ combineStr + "_eachPanel_"+str(substeps)+"substeps"
    steady_state_plots.plot_panels_vs_time_jwenn(model, day=daySel, combine_tubes = combineStr, savename = None)

    totalHead=model.receiver.pressure_drop_with_tower
    print('total head loss is %.4f' % totalHead)

    qsabs=getattr(model.results, 'abs_flux')

    Tw_outer=getattr(model.results, 'Tw')
    Tw_outer-=273.15    #convert to [C]

    Tf_K=getattr(model.results, 'Tf')

    Tw_inner_high=getattr(model.results, 'Tw_inner_high')

    Tw_inner_high-=273.15   #convert to [C]
    Tf=Tf_K-273.15  #convert to [C]
    
    Tw_inner_low=getattr(model.results, 'Tw_inner_low')
    Tw_inner_low-=273.15    #convert to [C]

    htubes=getattr(model.results, 'htubes')
    eta_receiver=getattr(model.results, 'eta_therm_avg')
    print('the average receiver efficiency is %0.4f' % eta_receiver)

    h_max_avg= np.mean(np.max(htubes[:,:,:,d],axis=0))   #the average maximum heat transfer coefficient at the design point temperature
    print("the unweighted average max tube htc @ design point is: %0.4f" % h_max_avg)


    npts=Tw_outer.shape[3]
    npan=model.receiver.Npanels
        
    names=["Tw_outer","Tw_inner_high","Tw_inner_low","nPanels","nTubesSim"]
    array_list=[Tw_outer,Tw_inner_high,Tw_inner_low,np.array(model.receiver.Npanels),np.array(model.receiver.ntubesim)]
    # helpers.np_to_json(names, array_list, "key_temp_timeseries.json")


    plt.rcParams.update({'font.size': 15})
    [fig,ax] = plt.subplots()
    colors = ['midnightblue', 'maroon', 'darkgreen', 'gray', 'rebeccapurple', 'saddlebrown', 'goldenrod', 'cornflowerblue', 'C1', 'C2']
    ### plot all the max dTs across receiver
    dTs=np.zeros((npan, model.receiver.ntubesim, model.receiver.disc.nz) )
    for p in range(model.receiver.Npanels):
        for t in range(model.receiver.ntubesim):
            for time in range(npts): #find the maximum deltaTs over all times
                single_Tw_outer=Tw_outer[:,p,t,time]
                single_Tw_inner_low=Tw_inner_low[:,p,t,time]
                single_Tw_inner_peak=Tw_inner_high[:,p,t,time]
                # print("this tube's max Tf is %0.10f" % np.max(single_Tw) )
                dT=single_Tw_outer-single_Tw_inner_low
                dTs[p,t,:]=np.maximum(dT,dTs[p,t,:])  #this overwrites previous timepoints' max if a new one is found
            index_color = p % len(colors)
            color = colors[index_color]
            # color = colors[p] if p < npan/2 else colors[npan-1-p]
            dTmax=np.max(dTs[p,t,:]) #takes the maximum along all axial positions for all timepoints
            ax.scatter(p,dTmax,color=color)

    ax.set_ylabel('$\Delta T$ [C]')
    ax.set_xlabel('panel')
    ax.set_xticks(range(npan))
    imgName = "./imgs/"+ 'deltaT_totals'
    # plt.savefig(imgName, dpi = 500) 
    plt.show()
    
    csvName = '/thermal_results/'+case_string
    print('saving dT file to:', csvName)
    imgName = case_string + 'max_dTs'
    steady_state_plots.get_maximum_dTs(model, plot=True,save_img=True,save_csv=True,imgName=imgName,csvName=csvName)
    
    # allTubeTempInfo.to_csv( ('allTubeTempInfo_30yr_ctr_dsgnPt.csv') )

    # # plot fluid temperature and inner wall temp for one tube at peak design point
    # panelNo = 0
    # tubeNo = 0
    d = model.find_design_point()
    print(d)
    tPt = d
    [fig2,ax2] = plt.subplots()
    panelNo = 7
    tubeNo = 0
    # for time in range(npts):
    #     # ax2.plot(range(Tw_outer.shape[0]),Tw_outer[:,panelNo,tubeNo,time],label='crown'+str(time) )
    #     # ax2.plot(range(Tw_inner_low.shape[0]),Tw_inner_low[:,panelNo,tubeNo,time],label='inner wall, min'+str(time) )
    #     # ax2.plot(range(Tw_inner_high.shape[0]),Tw_inner_high[:,panelNo,tubeNo,time],label='inner wall, max'+str(time) )
    #     tube_deltas=Tw_outer[:,panelNo,tubeNo,time]-Tw_inner_low[:,panelNo,tubeNo,time]
    #     ax2.plot(range(Tw_outer.shape[0]),tube_deltas,label='$\Delta T_{total}$ at:'+str(time))
    #     # ax2.plot(range(Tf.shape[0]),Tf[:,panelNo,tubeNo,time],label='fluid')
    
    # tPt = 6
    # tube_deltas=Tw_outer[:,panelNo,tubeNo,tPt]-Tw_inner_low[:,panelNo,tubeNo,tPt]
    # locDTmax2=np.argmax(tube_deltas)
    # print('max deltaT total for panel 7 is %0.1f'% np.max(tube_deltas))
    # print('representative fluid temeprature for panel 7 max is %0.1f'% Tf[locDTmax2,panelNo,tubeNo,tPt])
    nz=Tw_outer.shape[0]
    xAxis = np.linspace(0,nz-1,nz)/nz # range(Tw_outer.shape[0])/Tw_outer.shape[0]
    # ax2.plot(xAxis,tube_deltas)
    # ax2.set_ylabel('$\Delta T_{total}$ [C]')
    # ax2.set_xlabel('axial position, z/L')
    # # ax2.set_xticks(  range(0,Tw_outer.shape[0]+5,5))
    # # ax2.legend()
    # imgName = "./imgs/"+ 'deltaTProf_tube0_panel7@time'+str(tPt)
    # # plt.savefig(imgName, dpi = 500) 
    # plt.show()
    
    # panelNo = 7
    # tubeNo = 0
    # nz=Tw_outer.shape[0]
    # xAxis = np.linspace(0,nz-1,nz)/nz # range(Tw_outer.shape[0])/Tw_outer.shape[0]
    # [fig3,ax3] = plt.subplots()
    # ax3.plot(xAxis,Tw_outer[:,panelNo,tubeNo,tPt],label='$T_{crown}$')
    # ax3.plot(xAxis,Tw_inner_low[:,panelNo,tubeNo,tPt],label='$T_{f}$')
    # ax3.plot(xAxis,Tw_inner_high[:,panelNo,tubeNo,tPt],label='$T_{i,HS}$')
    # # ax3.plot(xAxis,Tf[:,panelNo,tubeNo,tPt],label='fluid')
    # ax3.set_ylabel('temperature [C]')
    # ax3.set_xlabel('axial position, z/L')
    # # ax3.set_xticks(  range(0,Tw_outer.shape[0]+5,5) )
    # ax3.legend(loc=4)
    # imgName = "./imgs/"+ 'axialtempProf_tube0_panel7@time'+str(tPt)
    # # plt.savefig(imgName, dpi = 500) 
    # plt.show()

    # ### plot the crown inner and crown outer and coldside inner
    # panelNo = 7
    # tubeNo = 0
    # nz=Tw_outer.shape[0]
    # # xAxis = np.linspace(0,nz-1,nz)/nz # range(Tw_outer.shape[0])/Tw_outer.shape[0]
    # [figTS,axTS] = plt.subplots()
    # axInd=36
    # axTS.plot(Tw_outer[axInd,panelNo,tubeNo,:],label='$T_{o,crown}$',color='r',marker='.',linestyle='--')
    # axTS.plot(Tw_inner_high[axInd,panelNo,tubeNo,:],label='$T_{i,crown}$',color='m',marker='.',linestyle=':')
    # axTS.plot(Tw_inner_low[axInd,panelNo,tubeNo,:],label='$T_{i,unheated}$',color='b',marker='.',linestyle='-')
    # axTS.set_ylim(300,535)
    # axTS.set_ylabel('temperature (C)')
    # axTS.set_xlabel('time (hrs)')
    # # ax3.set_xticks(  range(0,Tw_outer.shape[0]+5,5) )
    # axTS.legend(loc=4)
    # imgName = "./imgs/"+ 'tempTimeseries'+str(panelNo)+'p_'+str(tubeNo)+'t'
    # # plt.savefig(imgName, dpi = 500) 
    # plt.show()

    # panelNo = 0
    # tubeNo = 0
    # [fig4,ax4] = plt.subplots()
    # ax4.plot(xAxis,Tw_outer[:,panelNo,tubeNo,tPt],label='crown')
    # ax4.plot(xAxis,Tw_inner_low[:,panelNo,tubeNo,tPt],label='inner wall, min')
    # ax4.plot(xAxis,Tw_inner_high[:,panelNo,tubeNo,tPt],label='inner wall, max')
    # ax4.plot(xAxis,Tf[:,panelNo,tubeNo,tPt],label='fluid')
    # ax4.set_ylabel('temperature [C]')
    # ax4.set_xlabel('axial position, z/L')
    # # ax4.set_xticks(  range(0,Tw_outer.shape[0]+5,5) )
    # ax4.legend()
    # imgName = "./imgs/"+ 'axialHTC_tube0_panel0'
    # plt.savefig(imgName, dpi=500)
    # plt.show()

    # tubeNo = 0
    # panelNo = 0
    # [fig5,ax5] = plt.subplots()
    # ax5.plot(range( htubes.shape[0]),htubes[:,panelNo,tubeNo,tPt] )
    # ax5.set_ylabel('heat transfer coefficient [W/m^2-K]')
    # ax5.set_xlabel('axial node')
    # ax5.set_xticks( range(0,Tw_outer.shape[0]+5,5) )
    # plt.show()

    # tubeNo = 0
    # panelNo = 6
    # [fig5,ax5] = plt.subplots()
    # ax5.plot(range( htubes.shape[0]),htubes[:,panelNo,tubeNo,tPt] )
    # ax5.set_ylabel('heat transfer coefficient [W/m^2-K]')
    # ax5.set_xlabel('axial node')
    # ax5.set_xticks( range(0,Tw_outer.shape[0]+5,5) )
    # plt.show()

        
    # plot all ratios at design point
    dTs_cond=np.zeros((npan, model.receiver.ntubesim, model.receiver.disc.nz) )
    dTs_conv=np.zeros((npan, model.receiver.ntubesim, model.receiver.disc.nz) )
    ratio=0.4981 
    counter=0
    [fig6,ax6] = plt.subplots()
    colors = ['midnightblue', 'maroon', 'darkgreen', 'gray', 'rebeccapurple', 'saddlebrown', 'goldenrod', 'cornflowerblue', 'C1', 'C2']
    for p in range(model.receiver.Npanels):
        for t in range(model.receiver.ntubesim):
            for time in range(npts):
                single_Tw_outer=Tw_outer[:,p,t,time]
                single_Tw_inner_low=Tw_inner_low[:,p,t,time]
                single_Tw_inner_peak=Tw_inner_high[:,p,t,time]
                # print("this tube's max Tf is %0.10f" % np.max(single_Tw) )
                dT_cond=single_Tw_outer-single_Tw_inner_peak
                dT_conv=single_Tw_inner_peak-single_Tw_inner_low
                dTs_cond[p,t,:]=np.maximum(dT_cond,dTs_cond[p,t,:])    #finds the maximum over all times evaluated
                dTs_conv[p,t,:]=np.maximum(dT_conv,dTs_conv[p,t,:])
            index_color= p % len(colors)
            color = colors[index_color]
            # color = colors[p] if p < npan/2 else colors[npan-1-p]
            ax6.scatter(p,( np.max(dTs_cond[p,t,:])/np.max(dTs_conv[p,t,:]) ),color=color)
            counter+=( np.max(dTs_cond[p,t,:])/np.max(dTs_conv[p,t,:]) )
    avg_manually=counter/(model.receiver.Npanels*model.receiver.ntubesim)
    ax6.set_ylabel('$\Delta T_{cond} / \Delta T_{conv}$ @ noon')
    ax6.set_xlabel('panel')
    ax6.axhline(y=ratio, color = 'k', linestyle = '--', label='og average')
    ax6.axhline(y=avg_manually, color = 'b',linestyle = '--', label='ratio average')
    ax6.legend()
    ax6.set_xticks(range(npan))
    plt.show()

    # # plot select times
    # plt.rcParams.update({'font.size': 15})
    # npts_start=0 # starting index
    # npts=13       # ending index
    # dTs_cond=np.zeros((npan, model.receiver.ntubesim, model.receiver.disc.nz) )
    # dTs_conv=np.zeros((npan, model.receiver.ntubesim, model.receiver.disc.nz) )
    # # ratio=0.4981 
    # counter=0
    # [fig6,ax6] = plt.subplots()
    # colors = ['midnightblue', 'maroon', 'darkgreen', 'gray', 'rebeccapurple', 'saddlebrown', 'goldenrod', 'cornflowerblue', 'C1', 'C2']
    # for p in range(model.receiver.Npanels):
    #     for t in range(model.receiver.ntubesim):
    #         for time in range(npts_start,npts):
    #             single_Tw_outer=Tw_outer[:,p,t,time]
    #             single_Tw_inner_low=Tw_inner_low[:,p,t,time]
    #             single_Tw_inner_peak=Tw_inner_high[:,p,t,time]
    #             # print("this tube's max Tf is %0.10f" % np.max(single_Tw) )
    #             dT_cond=single_Tw_outer-single_Tw_inner_peak
    #             dT_conv=single_Tw_inner_peak-single_Tw_inner_low
    #             counter+=( np.max(dT_cond)/np.max(dT_conv) ) #assuming both maxes happen at same place
    #             dTs_cond[p,t,:]=np.maximum(dT_cond,dTs_cond[p,t,:])    #finds the maximum over all times evaluated
    #             dTs_conv[p,t,:]=np.maximum(dT_conv,dTs_conv[p,t,:])

    #         color = colors[p] if p < npan/2 else colors[npan-1-p] # doesn't work if too many panels
    #         ax6.scatter(p,( np.max(dTs_cond[p,t,:])/np.max(dTs_conv[p,t,:]) ),color=color)
    #         # counter+=( np.max(dTs_cond[p,t,:])/np.max(dTs_conv[p,t,:]) )
    # avg_manually=counter/(model.receiver.Npanels*model.receiver.ntubesim*npts)
    # ax6.set_ylabel('$\Delta T_{cond} / \Delta T_{conv}$')
    # ax6.set_xlabel('panel')
    # # ax6.axhline(y=ratio, color = 'k', linestyle = '--', label='predicted ratio')
    # ax6.axhline(y=avg_manually, color = 'b',linestyle = '--', label='average ratio')
    # # ax6.legend()
    # ax6.set_xticks(range(npan))
    # imgName = "./imgs/"+ 'deltaT_ratio'
    # # plt.savefig(imgName, dpi = 500) 
    # plt.show()

# ## plot the axial profiles on each tube
#     [fluxFig,fluxAx] = plt.subplots()
#     for p in range(model.receiver.Npanels):
#         for t in range(model.receiver.ntubesim):
#                 fluxAx.plot(xAxis,qsabs[:,p,t,d])
#     fluxAx.set_ylabel('flux $(W/m^2)$')
#     fluxAx.set_xlabel('axial position')
#     # imgName = "./imgs/"+ 'allTubeTempInfo_atMax.png'
#     # plt.savefig(imgName, dpi = 500) 
#     plt.show()

## plot the axial profiles on each tube
    [fluxFig2,fluxAx2] = plt.subplots()
    fluxAx2.plot(xAxis,qsabs[:,0,0,d])
    fluxAx2.set_ylabel('flux $(W/m^2)$')
    fluxAx2.set_xlabel('axial position')
    # imgName = "./imgs/"+ 'allTubeTempInfo_atMax.png'
    # plt.savefig(imgName, dpi = 500) 
    plt.show()
print('the number of tubes is:',model.receiver.ntubes_per_panel)
# print('flow path mass flows are:',branchModel.receiver.mflow_per_path)
# print('the total design point Q into fluid is:',branchModel.receiver.Qfluid)
print('the temperature rise in panel 0, tube 0 is:',Tf[-1,0,0,d]-Tf[0,0,0,d])
