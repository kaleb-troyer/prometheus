"""
created by jwenner on 6/11/25
---
collect data from results files and create twp summary files with strainrange and creep lookup tables for each temperature combo

edited by bepagel 7/11/25 - incorporated interpolation data points as well.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from srlife import library, solverparams, damage
import post_process_helpers as helpers
import os

#Run this file after running all cases through CHTC for that specific Campaign
#This file will put results together for crown_stress,max_stress, and creep damage
#This file will also interpolate any possible cases with 1D Motion i.e. if case doesn't exist but, to left and right cases exist then it will interpolate


##------------------------------------Beginning of User Inputs Section---------------------------------------------##
## read each file
material        ='A282'         # material and case_name will need changed for each simulation campaign
case_name       ='DMG_CP10_high_R'  #campaign name
substeps        = [30,25]    #substeps folders

## loop through each file
Tfs=list(range(275,600,25))   #Range of difference in crown temperature to fluid temperature
dTs=list(range(50,310,10))    #Range of fluid temepratures

ndays_select=4  # number of data days of interest
days=[1,2,3,4]  #days interested for parameter data (largest integer must be less than or equal to ndays_select integer value)


##----------------------------------End of User Inputs Section--------------------------------------------------------##

##-----------------------------Beginning of defining the Damage Material Model Section--------------------------------##
deformation_mat_string = 'base'
thermal_mat, deformation_mat, damage_mat = library.load_material(material, "base", deformation_mat_string, "base")
params = solverparams.ParameterSet()
damage_model = damage.TimeFractionInteractionDamage(params["damage"]) # some instance of metal with library properties
##-----------------------------End of Defining the Damage Material Model Section---------------------------------------##


##-----------------------------Beginning of Extracting Data from Results Section---------------------------------------##
strain_range_id_data=[]      #max crown strain range with indentication data
dc_damage_id_data=[]         #damage from creep at crown with identification data
comb_para_id=[]              #combined parameter with identification data
non_existing_pairs_data=[]   #tuple pairs of non-existing result files for [Tf,dT]


n_missing   =0    #number of missing results files
n_found     =0    #Number of results files found


##extract data from results
for Tf in Tfs:   
    for dT in dTs:
        filestrings=''
        for substep in substeps:
            model_name=case_name+f'_{Tf}Tf_{dT}dT'
            file_name=f'{model_name}_with_{material}_results.csv'
            file_string=f'{case_name}+/results_{substep}substeps/{file_name}'
            file_string=case_name+'/unprocessed_results/'+str(substep)+'substeps/'+file_name
            if os.path.exists(file_string):
                filestrings=file_string
                break
            else:
                continue
        
        try:

            results_df=pd.read_csv(filestrings)
            ## extract each day's maximum crown strainrange
            strainranges_at_crown=helpers.get_daily_strainranges_at_loc(results_df,ndays_select)

            ## extract each day's creep damage at the crown
            dcs_at_crown=helpers.get_daily_creep_damages_at_loc(results_df, damage_mat, material, ndays_select)
   
            strain_range_id_row=[strainranges_at_crown.tolist(),Tf,dT]
            dcs_at_crown_id_row=[dcs_at_crown.tolist(),Tf,dT]
            combined_parameters_id_row=[Tf,dT,strainranges_at_crown.tolist(),dcs_at_crown.tolist()]  

            strain_range_id_data.append(strain_range_id_row)
            dc_damage_id_data.append(dcs_at_crown_id_row)
            comb_para_id.append(combined_parameters_id_row)

            n_found+=1

        except FileNotFoundError:
            print(file_name,'not found')
            pair_error=(Tf,dT)
            non_existing_pairs_data.append(pair_error)
            n_missing+=1
##-----------------------------End of Extracting Data from Results File Section---------------------------------------------##

##------------------------------Beginning of Interpolating Data Section-----------------------------------------------------##
for pair in non_existing_pairs_data:
    non_Tf=pair[0]
    non_dT=pair[1]

    existing_horizontal_left=[]   #checks the point on the LEFT of the non-existing point to see if results exist
    existing_horizontal_right=[]  #checks the point on the RIGHT of the non-existing point to see if results exist
    existing_vertical_top=[]      #checks the point ABOVE the non-existing point to see if results exist
    existing_vertical_bottom=[]   #checks the point BELOW the non-existing point to see if results exist

    strain_range_data=[]
    dc_damage_at_crown_data=[]

    strain_range_interp_vals=[]
    dc_damage_at_crown_interp_vals=[]


    for idx in comb_para_id:
        if idx[0]==(non_Tf+25) and idx[1]==non_dT:   #Above non-existing point case
            existing_vertical_top.append(idx)
        elif idx[0]==(non_Tf-25) and idx[1]==non_dT:  #Below non-existing point case
            existing_vertical_bottom.append(idx)
        elif idx[0]==non_Tf and idx[1]==(non_dT+10): #to the right of non-existing point case
            existing_horizontal_right.append(idx)
        elif idx[0]==non_Tf and idx[1]==(non_dT-10): #to the left of the non-existing point case
            existing_horizontal_left.append(idx)
        else:
            continue
    
    ##BOTH HORIZONTAL AND VERTICAL INTERPOLATION
    if len(existing_vertical_bottom)==1 and len(existing_vertical_top)==1 and len(existing_horizontal_left)==1 and len(existing_horizontal_right)==1:
        strain_range_data_h=[]    #for both case
        strain_range_data_v=[]    #for both case
        dc_damage_at_crown_data_h=[]   #for both case
        dc_damage_at_crown_data_v=[]   #for both case

        strain_v1=[]   #for plotting existing strain range for vertical top case
        strain_v2=[]   #for plotting existing strain range for vertical bottom case
        creep_v1=[]    #for plotting existing creep damage for vertical top case
        creep_v2=[]    #for plotting existing creep damage for vertical bottom case
        
        strain_h1=[]   #for plotting existing strain range for horizontal left case
        strain_h2=[]   #for plotting existing strain range for horizontal right case
        creep_h1=[]    #for plotting existing creep damage for horizontal left case
        creep_h2=[]    #for plotting existing creep damage for horizontal right case

        horizontal_strain_range_interp_with_day_pairs=[]#####
        horizontal_dc_damage_at_crown_interp_with_day_pairs=[]#####
        vertical_strain_range_interp_with_day_pairs=[]####
        vertical_dc_damage_at_crown_interp_with_day_pairs=[]####

        ##horizontal case
        dT_existing=[existing_horizontal_left[0][1],existing_horizontal_right[0][1]]  
        for a in enumerate(days):
            strain_range_existing_day=[existing_horizontal_left[0][2][a[0]],existing_horizontal_right[0][2][a[0]]]
            dc_damage_at_crown_existing_day=[existing_horizontal_left[0][3][a[0]],existing_horizontal_right[0][3][a[0]]]
            strain_h1.append((existing_horizontal_left[0][2][a[0]],days[a[0]]))
            strain_h2.append((existing_horizontal_right[0][2][a[0]],days[a[0]]))
            creep_h1.append((existing_horizontal_left[0][3][a[0]],days[a[0]]))
            creep_h2.append((existing_horizontal_right[0][3][a[0]],days[a[0]]))
            strain_range_data_h.append(strain_range_existing_day)
            dc_damage_at_crown_data_h.append(dc_damage_at_crown_existing_day)
        for b in enumerate(strain_range_data_h):  #interpolation
            strainrange_l_interp=np.interp(non_dT,dT_existing,b[1])
            strain_range_interp_vals.append(strainrange_l_interp)
            horizontal_strain_range_interp_with_day_pairs.append((strainrange_l_interp,days[b[0]])) ####
        for c in enumerate(dc_damage_at_crown_data_h):
            dc_damage_at_crown_l_interp=np.interp(non_dT,dT_existing,c[1])
            dc_damage_at_crown_interp_vals.append(dc_damage_at_crown_l_interp) 
            horizontal_dc_damage_at_crown_interp_with_day_pairs.append((dc_damage_at_crown_l_interp,days[c[0]]))######

        ##vertical case
        Tf_existing=[existing_vertical_bottom[0][0],existing_vertical_top[0][0]]
        for d in enumerate(days):
            strain_range_existing_day=[existing_vertical_top[0][2][d[0]],existing_vertical_bottom[0][2][d[0]]]
            dc_damage_at_crown_existing_day=[existing_vertical_top[0][3][d[0]],existing_vertical_bottom[0][3][d[0]]]
            strain_v1.append((existing_vertical_top[0][2][d[0]],days[d[0]]))
            strain_v2.append((existing_vertical_bottom[0][2][d[0]],days[d[0]]))
            creep_v1.append((existing_vertical_top[0][3][d[0]],days[d[0]]))
            creep_v2.append((existing_vertical_bottom[0][3][d[0]],days[d[0]]))
            strain_range_data_v.append(strain_range_existing_day)
            dc_damage_at_crown_data_v.append(dc_damage_at_crown_existing_day)
        for e in enumerate(strain_range_data_v):
            strainrange_i_interp=np.interp(non_Tf,Tf_existing,e[1])
            strain_range_interp_vals.append(strainrange_i_interp)
            vertical_strain_range_interp_with_day_pairs.append((strainrange_i_interp,days[e[0]]))####
        for f in enumerate(dc_damage_at_crown_data_v):
            dc_damage_at_crown_i_interp=np.interp([non_Tf],Tf_existing,f[1])
            dc_damage_at_crown_interp_vals.append(dc_damage_at_crown_i_interp)
            vertical_dc_damage_at_crown_interp_with_day_pairs.append((dc_damage_at_crown_i_interp,days[f[0]]))#####

        h1_existing_strain,h1_existing_s_day=zip(*strain_h1)
        h2_existing_strain,h2_existing_s_day=zip(*strain_h2)
        v1_existing_strain,v1_existing_s_day=zip(*strain_v1)
        v2_existing_strain,v2_existing_s_day=zip(*strain_v2)

        h1_creep_damage,h1_existing_cd_day=zip(*creep_h1)
        h2_creep_damage,h2_existing_cd_day=zip(*creep_h2)
        v1_creep_damage,v1_existing_cd_day=zip(*creep_v1)
        v2_creep_damage,v2_existing_cd_day=zip(*creep_v2)

        h_strain_value,h_strain_day=zip(*horizontal_strain_range_interp_with_day_pairs)
        h_creep_damage_value,h_creep_damage_day=zip(*horizontal_dc_damage_at_crown_interp_with_day_pairs)
        v_strain_value,v_strain_day=zip(*vertical_strain_range_interp_with_day_pairs)
        v_creep_damage_value,v_creep_damage_day=zip(*vertical_dc_damage_at_crown_interp_with_day_pairs)


        fig,ax=plt.subplots(tight_layout=True)
        ax.plot(h1_existing_s_day,h1_existing_strain,marker='.',linestyle='-',color='black',label="Existing Horizontal Left Case")
        ax.plot(h2_existing_s_day,h2_existing_strain,marker='.',linestyle='-',color='orange',label="Existing Horizontal Right Case")
        ax.plot(v1_existing_s_day,v1_existing_strain,linestyle=':',color="blue",label='Existing Vertical Top Case')
        ax.plot(v2_existing_s_day,v2_existing_strain,linestyle=':',markersize=20,color="green",label='Existing Vertical Bottom Case')
        ax.plot(h_strain_day,h_strain_value,marker='*',linestyle='-',color='cyan',label="Horizontal Interpolation")
        ax.plot(v_strain_day,v_strain_value,marker='o',linestyle=':',color='magenta',label="Vertical Interpolation")
        plt.grid(True)
        ax.set_xlabel('Day')
        ax.set_ylabel('Maximum Strain Range at Crown')
        ax.legend()
        ax.set_title("Interpolation of Vertical and Horizontal Directions for (dT= "+str(non_dT)+" °C , Tf= "+str(non_Tf) +" °C)",fontsize=9)
        fig.savefig(dpi=300,fname=case_name+'_Strain_Range_Interpolation_Checker_plot')
        plt.show()
        plt.close()

        fig,ax=plt.subplots(tight_layout=True)
        ax.plot(h1_existing_cd_day,h1_creep_damage,marker='.',linestyle='-',color='black',label="Existing Horizontal Left Case")
        ax.plot(h2_existing_cd_day,h2_creep_damage,marker='.',linestyle='-',color='orange',label="Existing Horizontal Right Case")
        ax.plot(v1_existing_cd_day,v1_creep_damage,marker='.',linestyle=':',color="blue",label='Existing Vertical Top Case')
        ax.plot(v2_existing_cd_day,v2_creep_damage,marker='.',linestyle=':',color="green",label='Existing Vertical Bottom Case')
        ax.plot(h_creep_damage_day,h_creep_damage_value,linestyle='-',marker='*',color='cyan',label="Horizontal Interpolation")
        ax.plot(v_creep_damage_day,v_creep_damage_value,marker='*',linestyle=':',color='magenta',label="Vertical Interpolation")
        plt.grid(True)
        ax.set_xlabel('Day')
        ax.set_ylabel('Creep Damage at Crown')
        ax.legend()
        ax.set_title(" Interpolation of Vertical and Horizontal Directions for (dT= "+str(non_dT)+" °C , Tf= "+str(non_Tf) +" °C)",fontsize=9)
        fig.savefig(dpi=300,fname=case_name+'_Creep_Damage_Interpolation_Checker_plot')
        plt.show()
        plt.close()


    ##HORIZONTAL INTERPOLATION ONLY
    elif len(existing_horizontal_left)==1 and len(existing_horizontal_right)==1:   
        dT_existing=[existing_horizontal_left[0][1],existing_horizontal_right[0][1]]
        for g in enumerate(days):
            strain_range_existing_day=[existing_horizontal_left[0][2][g[0]],existing_horizontal_right[0][2][g[0]]]
            dc_damage_at_crown_existing_day=[existing_horizontal_left[0][3][g[0]],existing_horizontal_right[0][2][g[0]]]
            strain_range_data.append(strain_range_existing_day)
            dc_damage_at_crown_data.append(dc_damage_at_crown_existing_day)
        for h in strain_range_data:
            strainrange_i_interp=np.interp(non_dT,dT_existing,h)
            strain_range_interp_vals.append(strainrange_i_interp)
        for i in dc_damage_at_crown_data:
            dc_damage_at_crown_i_interp=np.interp(non_dT,dT_existing,i)
            dc_damage_at_crown_interp_vals.append(dc_damage_at_crown_i_interp)


    ##VERTICAL INTERPOLATION ONLY
    elif len(existing_vertical_top)==1 and len(existing_vertical_bottom)==1:   #VERTICAL INTERPOLATION ONLY
        Tf_existing=[existing_vertical_bottom[0][0],existing_vertical_top[0][0]]
        for j in enumerate(days):
            strain_range_existing_day=[existing_vertical_top[0][2][j[0]],existing_vertical_bottom[0][2][j[0]]]
            dc_damage_at_crown_existing_day=[existing_vertical_top[0][3][j[0]],existing_vertical_bottom[0][2][j[0]]]
            strain_range_data.append(strain_range_existing_day)
            dc_damage_at_crown_data.append(dc_damage_at_crown_existing_day)
        for k in strain_range_data:
            strainrange_i_interp=np.interp(non_Tf,Tf_existing,k)
            strain_range_interp_vals.append(strainrange_i_interp)
        for l in dc_damage_at_crown_data:
            dc_damage_at_crown_i_interp=np.interp(non_Tf,Tf_existing,l)
            dc_damage_at_crown_interp_vals.append(dc_damage_at_crown_i_interp)

    else:
        strain_range_NaN=[]  #Replace non-existing (dT,Tf) pairs with the value NaN for each strain df
        dc_damage_NaN=[]     #Replace non-existing (dT,Tf) pairs with the value NaN for each creep damage df
        for index in range(ndays_select):
            strain_range_value=float('nan')
            dc_damage_value=float('nan')
            strain_range_NaN.append(strain_range_value)
            dc_damage_NaN.append(dc_damage_value)
        strain_range_id_data.append([strain_range_NaN,non_Tf,non_dT])
        dc_damage_id_data.append([dc_damage_NaN,non_Tf,non_dT])
        continue

    strain_range_id_data.append([float(np.mean(strain_range_interp_vals)),non_Tf,non_dT])  #add interpolated strain range to master strain range list
    dc_damage_id_data.append([float(np.mean(dc_damage_at_crown_interp_vals)),non_Tf,non_dT])  #add interpolated strain range to master dc creep at crown list


#-----------------------------------#
sorted_strain_range_id_data=sorted(strain_range_id_data,key=lambda x:(x[1],x[2])) # sort the strain range identification data with interpolated values

sorted_dc_damage_at_crown_id_data=sorted(dc_damage_id_data,key=lambda x:(x[1],x[2]))  #sort the dc creep damage at crown identification data with interpolated values
##---------------------------------------------------End of Interpolation Section----------------------------------------------##


##---------------------------------Beginning of Writing CSV files for Strain Range and Creep Damage Section--------------------##
## creates CSV files for strain range data
for index,m in enumerate(days):
    day=m
    strain_range_data=[]
    for idx in sorted_strain_range_id_data:
        daily_strain_range=idx[0][index]
        Tf=idx[1]
        dT=idx[2]
        daily_strain_row=[daily_strain_range,Tf,dT]
        strain_range_data.append(daily_strain_row)
    strain_df=pd.DataFrame(strain_range_data,columns=['strainrange_crown','Tf','dT'])
    strain_file_name='strainrange_crown_day'+str(day)+'_'+case_name+'.csv'
    strain_file_path=case_name+'/processed_results/'+strain_file_name        #FILE PATH
    strain_df.to_csv(strain_file_path,index=False)
    
## creates CSV files for creep damage at crown data
    dc_damage_data=[]
    for idx in sorted_dc_damage_at_crown_id_data:
        daily_dc_damage=idx[0][index]
        Tf=int(idx[1])
        dT=int(idx[2])
        daily_cd_damage_row=[daily_dc_damage,Tf,dT]
        dc_damage_data.append(daily_cd_damage_row)

    dc_damage_df=pd.DataFrame(dc_damage_data,columns=['creep_damage_crown','Tf','dT'])
    dc_damage_file_name='creep_damage_crown_day'+str(day)+'_'+case_name+'.csv'   
    dc_damage_file_path=case_name+'/processed_results/'+dc_damage_file_name   #FILE PATH
    dc_damage_df.to_csv(dc_damage_file_path,index=False)

print(f"{n_missing} missing files")
print(f"{n_found} files found")
##-----------------------------End of Writing CSV files for Strain Range and Creep Damage Section--------------------------##


##-----------------------------Beginning of Extrapolating Creep Damage Section---------------------------------------------##

### extrapolate and save the 20th creep damage in a separate file
dc20_damage_data=[]
for run in sorted_dc_damage_at_crown_id_data:
    # get the 20th day creep damage
    dcs =run[0]
    Tf  =run[1]
    dT  =run[2]
    dc20=helpers.extrap_20th_day_creep_damage(dcs)
    # append to master list
    dc_damage_row=[dc20,Tf,dT]
    dc20_damage_data.append(dc_damage_row)

    # ##### visually check the creep damage extrapolation & data
    # fig,ax=plt.subplots(tight_layout=True)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # # plt.grid(True)
    # ax.scatter([1,2,3,4],dcs)
    # ax.scatter([20],dc20)
    # ax.set_xlabel('# cycles')
    # ax.set_ylabel('creep damage')
    # ax.grid(color='gray',axis='both',which='both',alpha=0.3)
    # plt.show()
    # #####

# save in df
dc_damage_df=pd.DataFrame(dc20_damage_data,columns=['dc20_crown','Tf','dT'])
dc_damage_file_name='creep_damage_crown_day20_'+case_name+'.csv'
dc_damage_file_path=case_name+'/processed_results/'+dc_damage_file_name
dc_damage_df.to_csv(dc_damage_file_path,index=False)
##----------------------------------End of Extrapolating Creep Damage Section------------------------------------------##