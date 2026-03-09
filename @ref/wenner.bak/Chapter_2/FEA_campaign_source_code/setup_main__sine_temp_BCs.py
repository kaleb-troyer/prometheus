"""
created on 5/15/2025, updated 6/5/2025
@author: jwenner
@last edited: bepagel 06/17/2025 --added append model feature to append model to model list for CHTC --also imported os
"""

  
import numpy as np
import matplotlib.pyplot as plt
import sys
import BoundaryConditions as BC
import pandas as pd
sys.path.append('../..')
from srlife import receiver
import seaborn as sns
import os


# Setup the base receiver
if __name__ == "__main__":
  casename='test_case_script' # change this for each campaign you run
  inputs_df=pd.read_csv("dataframes/case_files/"+casename+'.csv')
  ## read case file
  R     =inputs_df['R'].values[0]               # ratio of conduction temperature difference to convection temperature difference. Original from crescent dunes: 0.5
  p_max =inputs_df['p_max'].values[0]           # (MPa) inner pressure diameter
  REF   =inputs_df['REF'].values[0]             # refinement factor

  # operating times
  period    =inputs_df['period'].values[0]         # (hrs) cycle period
  days      =inputs_df['days'].values[0]           # (-) number of cycles represented in the problem 
  t_ph      =inputs_df['t_ph'].values[0]           # (hrs)
  t_op      =inputs_df['t_op'].values[0]           # (hrs)
  substeps  =inputs_df['substeps'].values[0]       # (1/hrs) 


  # Basic receiver geometry
  d_outer   =inputs_df['d_outer'].values[0]        # (mm)
  r_outer   =d_outer/2                             # (mm)
  thickness =inputs_df['thickness'].values[0]      # (mm)
  height    =inputs_df['height'].values[0]         # (mm). Doesn't really matter - no axial variance in boundary temps and we take a 2D slice

  ## start the loop
  dTs=(190,)        # (C) total temperature difference at the crown
  Tfs=(425,)        # (C) fluid temperature 
  for Tf in Tfs:
    for dT in dTs:
        dT_cond     =dT*(1-1/(1+R))
        dT_conv     =dT/(1+R)
        T_outer_peak=Tf+dT                # the crown temperature on outer surface
        T_inner_peak=T_outer_peak-dT_cond # the crown temperature on inner surface
        T_outer_back=T_inner_peak-dT_conv # cool side of tube

        t_ramp_P  =t_ph       # pressure rampup time
        t_shutdown=t_op+t_ph  # time when shutdown starts
               
        # Time increments throughout the 24 hour day
        DaySteps=substeps*period #predicts the array length of one day's ntimes
        times = np.linspace(0,period*days,int(period*days)*substeps+1)
        filename=casename+f"_{Tf}Tf_{dT}dT"
        
        # remove the T_base because it is not time varying. Easier to make BCs with only time varying components
        T_base                =30 # C
        Tf_time_vary          =Tf-T_base
        T_outer_peak_time_vary=T_outer_peak-T_base

        #tube mesh discretization
        nr =3*REF   # 6 nodes for 1.25 th (3*REF)
        nt =72*REF  #
        nz =30      # nominally 30

        # create inner wall, bottom: no change in Tf the entire day, surface is isothermal
        fFracArrayInnerBottom =BC.buildFracNormal(times[:int(DaySteps+1)],t_ph,t_op)
        fArrayInnerBottom     =Tf_time_vary*fFracArrayInnerBottom
        innerTempArrayBottom  =BC.buildSurfUniTemp(fArrayInnerBottom,np.ones(int(nt/2)),nz)

        # create inner wall, top: the deltaT_conv will grow throughout the day as flux increases
        fFracArrayInnerDelta  =BC.buildDeltaTFracCos_wPH(times[:int(DaySteps+1)],t_op,t_ph)
        innerTempArrayTop     =BC.buildSurfCosine(fArrayInnerBottom,BC.makeThetaArray(int(nt/2)),fFracArrayInnerDelta,dT_conv,int(nt/2),nz)

        # combine top and bottom for total inner wall array day 1
        innerTempArrayDay1    =np.concatenate( (innerTempArrayTop,innerTempArrayBottom) ,axis=1) 

        innerTempArrayDayN=np.tile(innerTempArrayDay1[1:int(DaySteps+1)],(int(days-1),1,1)) #generate array for all days after the first day
        innerTempArray_unbased=np.concatenate( (innerTempArrayDay1,innerTempArrayDayN), axis=0 ) # combine the N days with the 1st day

        # create outer top and bottom surface temperatures
        outerTempArrayTop     =BC.buildSurfCosine(fArrayInnerBottom,BC.makeThetaArray(int(nt/2)),fFracArrayInnerDelta,dT,int(nt/2),nz ) #spatial variance imposed on existing time variance
        outerTempArrayBottom  =BC.buildSurfUniTemp(fArrayInnerBottom,np.ones(int(nt/2)),nz)

        # create full outer surface for day 1, N days, and combine them
        outerTempArrayDay1    =np.concatenate( (outerTempArrayTop,outerTempArrayBottom) ,axis=1) 
        outerTempArrayDayN    =np.tile(outerTempArrayDay1[1:int(DaySteps+1)],(int(days-1),1,1)) #replicate the daily outerTempArray for the number of days to simulate
        outerTempArray_unbased=np.concatenate( (outerTempArrayDay1,outerTempArrayDayN), axis=0 ) 

        
        # now that time variance is taken care of, can re-add the time-independent temperature base
        innerTempArray_C=innerTempArray_unbased+T_base
        outerTempArray_C=outerTempArray_unbased+T_base

        # make pressure boundary condition, extend to N days
        pressureArrayDay1 =BC.pressure(p_max, times[:int(DaySteps+1)],t_ramp_P,t_shutdown) #generates an array of dimension: time 
        pressureArrayDayN =np.tile(pressureArrayDay1[1:],int(days-1) )
        pressureArray     =np.concatenate( (pressureArrayDay1,pressureArrayDayN) ) 



        ## plotting checks at desired times, comment/uncomment as desired
        time_check_1=int(np.where(times[:]==1)[0]) 
        time_check_2=int(np.where(times[:]==7)[0]) 
        # time_check_3=int(np.where(times[:]==12)[0])

        # # these heatmaps unroll the inner and outer surfaces and plot them
        # sns.heatmap(outerTempArray_C[time_check_1].T,cbar_kws={'label': 'temperature [C]'}) #check that I did it right...
        # plt.ylabel("z dimension [nodes]")
        # plt.xlabel("radial position [nodes]")
        # plt.title("Outer Wall Temperature at "+str(time_check_1/substeps)+' hours') # title - BP
        # plt.show()

        # sns.heatmap(innerTempArray_C[time_check_1].T,cbar_kws={'label': 'temperature [C]'}) #check that I did it right...
        # plt.ylabel("z dimension [nodes]")
        # plt.xlabel("radial position [nodes]")
        # plt.title("Inner Wall Temperature at "+str(time_check_1/substeps)+' hours') # title - BP
        # plt.show()

        # sns.heatmap(outerTempArray_C[time_check_2].T,cbar_kws={'label': 'temperature [C]'}) #check that I did it right...
        # plt.ylabel("z dimension [nodes]")
        # plt.xlabel("radial position [nodes]")
        # plt.title("Outer Wall Temperature at "+str(time_check_2/substeps)+' hours') # title - BP
        # plt.show()

        # sns.heatmap(innerTempArray_C[time_check_2].T,cbar_kws={'label': 'temperature [C]'}) #check that I did it right...
        # plt.ylabel("z dimension [nodes]")
        # plt.xlabel("radial position [nodes]")
        # plt.title("Inner Wall Temperature at "+str(time_check_2/substeps)+' hours') # title - BP
        # plt.show()
        
        # sns.heatmap(outerTempArray_C[time_check_3].T,cbar_kws={'label': 'temperature [C]'}) #check that I did it right...
        # plt.ylabel("z dimension [nodes]")
        # plt.xlabel("radial position [nodes]")
        # plt.title("Outer Wall Temperature at "+str(time_check_3/substeps)+' hours') # title - BP
        # plt.show()

        # sns.heatmap(innerTempArray_C[time_check_3].T,cbar_kws={'label': 'temperature [C]'}) #check that I did it right...
        # plt.ylabel("z dimension [nodes]")
        # plt.xlabel("radial position [nodes]")
        # plt.title("Inner Wall Temperature at "+str(time_check_3/substeps)+' hours') # title - BP
        # plt.show()

        # plot surface temperature profiles
        fontsize=15
        zNode   =10
        dtheta  =360/nt
        thetas  =dtheta*np.linspace( int(nt/4),int(3*nt/4),int(nt/2) )
        fig,ax  =plt.subplots(tight_layout=True)
        ax.set_xlabel('circumferential position $(^{\circ})$',fontsize=fontsize)
        ax.set_ylabel('temperature (C)',fontsize=fontsize)
        ax.plot(thetas,outerTempArray_C[time_check_1,int(nt/4):int(3*nt/4),zNode],label='outer wall at '+str(time_check_1/substeps)+' (hrs)',color='black',linestyle='-')
        ax.plot(thetas,innerTempArray_C[time_check_1,int(nt/4):int(3*nt/4),zNode],label='inner wall at '+str(time_check_1/substeps)+' (hrs)',color='black',linestyle='--')
        ax.plot(thetas,outerTempArray_C[time_check_2,int(nt/4):int(3*nt/4),zNode],label='outer wall at '+str(time_check_2/substeps)+' (hrs)',color='black',linestyle=':')
        ax.plot(thetas,innerTempArray_C[time_check_2,int(nt/4):int(3*nt/4),zNode],label='inner wall at '+str(time_check_2/substeps)+' (hrs)',color='black',linestyle='-.')
        ax.legend(fontsize=fontsize)
        plt.yticks(fontsize=fontsize); plt.xticks(fontsize=fontsize)
        # plt.savefig( ( 'imgs/'+noteString+'angularDesc_'+str(timeCheck1)+'&'+str(timeCheck2)+'BCs'), dpi='figure', format='png')
        plt.show()
        plt.close()
        

        ## plot temperature timeseries
        fig,ax1 = plt.subplots(tight_layout=True)
        ax1.plot(times, innerTempArray_C[:,0,0], color='black', label='$T_{i,unheated}$',linestyle=':')
        ax1.plot(times, outerTempArray_C[:,int(nt/4),-1], color='black', label='$T_{o,crown}$',linestyle='--')
        ax1.plot(times, innerTempArray_C[:,int(nt/4),-1], color='black', label='$T_{i,crown}$',linestyle='-')

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.legend(loc=8,fontsize=fontsize)

        # make a ytick labeling based on desired step
        end=times[-1]
        step=6
        ax1.set_xticks(range(0,int(end)+step,step))

        ax1.set_xlabel('time (hr)', fontsize=fontsize)
        ax1.set_ylabel('temperature (C)',fontsize=fontsize)

        # also overlay pressure
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'black'
        ax2.plot(times, pressureArray, color=color, label='$P_{i}$')
        ax2.set_ylabel('pressure (MPa)', color=color, fontsize=fontsize)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc=1,fontsize=fontsize)
        
        plt.yticks(fontsize=fontsize); plt.xticks(fontsize=fontsize)
        # plt.savefig( ( '/imgs/'+noteString+'BCs'), dpi='figure', format='png')
        plt.show()
        plt.close()

        #Convert all temps to Kelvin
        innerTempArray_K=innerTempArray_C+273.15 #K
        outerTempArray_K=outerTempArray_C+273.15 #K
        T_base_K        =T_base+273.15 #K



        ## setting up the model with the generated boundary conditions
        # make the model instance
        panel_stiffness ="disconnect" # Panels are disconnected from one another
        model           =receiver.Receiver(period, days, panel_stiffness)

        # setup a panel with one tube
        tube_stiffness ="disconnect"
        panel_0        =receiver.Panel(tube_stiffness)
       
        # Setup a tube and assign it to the correct panel - ENSURE TEMPS ARE IN KELVIN
        tube_0 =receiver.Tube(r_outer, thickness, height, nr, nt, nz, T0 = T_base_K)
        tube_0.set_times(times)
        tube_0.set_bc(receiver.FixedTempBC(r_outer-thickness,height, nt, nz, times, innerTempArray_K), "inner")
        tube_0.set_bc(receiver.FixedTempBC(r_outer, height,nt, nz, times, outerTempArray_K), "outer")
        tube_0.set_pressure_bc(receiver.PressureBC(times, pressureArray))

        # Assign to panel 0
        panel_0.add_tube(tube_0)        
        model.add_panel(panel_0)
        


        ### Save the receiver to an HDF5 file        
        print(filename)
        model.save("srlife_models/"+filename+".hdf5")


        #for CHTC ModelList.txt SUBMISSION 
        modelList_Path="CHTC_Folder/modelList.txt"
        if os.path.exists(modelList_Path):
          pass
        else:
          with open(modelList_Path, "x") as file:
            modelList_Path.close()
        with open(modelList_Path, "a") as file:
          file.write(casename+'_'+str(Tf)+'Tf_'+str(dT)+'dT\n')
