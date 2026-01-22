"""
created by: jwenner on 7/15/2025
--
purpose: makes two table files (currently...)
            DMG_table_relaxed_crown_model   - a table that contains epsilon, 20th day creep damage for each temperature case. Un-interpolated and for use by SRLIFE-capable setups
            LTE_table_relaxed_crown_model   - a table containing LTEs for each temperature case. Potentially higher resolution than original data. Used for damage map and accessible by non-SRLIFE-capable setups
            damage_map                      - a visualization of how creep-fatigue damage changes according to temperature case. Uses LTE_table for now.
"""

import pandas as pd
from srlife import solverparams, library, damage
import post_process_helpers as helpers
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sb
import plot_damage_map as plotter


## state some assumptions
frac_op_red=9/12     # original cycle uses sine waves which can be scaled for different 
print(f'assuming actual cycle length is {frac_op_red} of that simulated')
days_in_year=365
print('assuming',days_in_year,'operational days per year')


## input the case name and material string
case_name = 'DMG_CP1_norm_R'
mat_string = 'A230'


## make SRLIFE damage model
deformation_mat_string                      = 'base' 
thermal_mat, deformation_mat, damage_mat    = library.load_material(mat_string, "base", deformation_mat_string, "base")
params                                      = solverparams.ParameterSet()


damage_model                = damage.TimeFractionInteractionDamage(params["damage"]) # some instance of metal with library properties
damage_model.extrapolate    ='last'
damage_model.order          =3
print('extrapolation method is:', damage_model.extrapolate,' order is:', damage_model.order)  #I found that my parameter settings weren't being implemented so now i always check


## load the selected srlife results, get strainrange and creep damage
strainrange_filestring =case_name+'/processed_results/strainrange_crown_day4_'+case_name+'.csv'   #strainrange_crown_day 4
dframe_dy4_strainranges = pd.read_csv(strainrange_filestring)


dc_filestring = case_name+'/processed_results/creep_damage_crown_day20_'+case_name+'.csv'   #creep damage day 20
dframe_dy20_dcs = pd.read_csv(dc_filestring)


DMG_dframe_parent=pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None) #parent dataframe that I append to
# strainrange_dframe_parent=pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None) #parent dataframe that I append to


## process the strainranges into fatigue damages, get fatigue df, also add the creep damage
## also calculate/add LTE, damage ratio
Tfs=list(range(275,600,25))
dTs=list(range(50,310,10))


for Tf in Tfs:
    for dT in dTs:
        # for 740H results ONLY since temperature is out of range
        if Tf==575 and (dT==280 or dT==290 or dT==300):
            print(f'skipping Tf={575},dT={dT}')
            continue
        T_crown=Tf+dT
        strainrange=dframe_dy4_strainranges.strainrange_crown[(dframe_dy4_strainranges.Tf==Tf) & (dframe_dy4_strainranges.dT==dT)].values # returns a single element, np array
        # get fatigue damage associated with one cycle
        df = 1/helpers.cycles_to_fail_lawless(damage_mat, T_crown, strainrange, cutoff_bool=False) #input temperature should be in C
        # get creep damage associated with one cycle
        dc = dframe_dy20_dcs.dc20_crown[(dframe_dy20_dcs.Tf==Tf)&(dframe_dy20_dcs.dT==dT)].values
        # account for assumed length of cycle compared to simulated cycle
        dc=dc*frac_op_red
        # now calculate lte cycles based on fatigue and creep damage
        lte_cycles=damage_model.calculate_max_cycles(damage_model.make_extrapolate( dc ), damage_model.make_extrapolate( df ), damage_mat, rep_min=1, rep_max=1e6)
        # get the end of life ratio of fatigue to creep damage
        if math.isinf(lte_cycles)==True:
            ratio_dmg = np.array([float("nan")])
        else:
            dc_sum = damage_model.make_extrapolate( dc )(lte_cycles)
            df_sum = damage_model.make_extrapolate( df )(lte_cycles)
            ratio_dmg = df_sum/dc_sum 
        # add all damage data in a single dataframe for srlife-users
        DMG_dframe_child = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None) 
        DMG_dframe_child['dT']          =np.array([dT])
        DMG_dframe_child['Tf']          =np.array([Tf])
        DMG_dframe_child['strainrange'] =strainrange
        DMG_dframe_child['df4']         =df
        DMG_dframe_child['dc20']        =dc
        DMG_dframe_child['lte_cycles']  =np.array([lte_cycles])
        DMG_dframe_child['ratio_dmgs']  =np.array([ratio_dmg])
        DMG_dframe_parent=pd.concat([DMG_dframe_parent,DMG_dframe_child],ignore_index=True)


## save the DMG_table using raw data
filestring_dmg_table=f'dataframes/DMG_table_relaxed_crown_model_{case_name}.csv'
DMG_dframe_parent.to_csv(filestring_dmg_table)


## create the LTE_table by using SRLIFE LTE extrapolation method with original data resolution
df_LTE = DMG_dframe_parent.pivot(index="Tf",columns='dT',values='lte_cycles') #re-organize data 
df_LTE = df_LTE.sort_index(ascending=False) #flip it


# save the table
filestring_lte_table=f'dataframes/LTE_table_relaxed_crown_model_{case_name}.csv'
df_LTE.to_csv(filestring_lte_table)


## plot the LTE table using plot_damage_map
savename=case_name
plotter.plot(DMG_dframe_parent, mat_string, days_in_year,savename)