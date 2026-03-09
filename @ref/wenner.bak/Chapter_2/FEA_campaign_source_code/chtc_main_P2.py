"""
Created on Monday 6/5/2025

@author: jwenner

P3 changes: -removed several unused functions
            -vastly enhanced data logging
            -solving settings are unchanged
"""
import numpy as np
import pandas as pd
import os
import sys
from srlife import receiver, solverparams, library, thermal, structural, system, damage, managers #damage_rainflow #no damage_rainflow module found

def get_quad_vm_max_results(model, vm_strainranges,vm_stresses):
    """
    purpose: get the maximum vm strain and the maximum vm stress indices
    ---
    model: objects with results and simulation info
    vm_strainranges:     list  - (ntimes x elements x 4) processed results in quadrature form
    vm_stresses:    array - (ntimes x elements x 4) processed results in quadrature form
    --- 
    max_stress_indices: list (ntimes x element x quad)
    max_stresses: list (ntimes x 1)
    max_strainrange_indices & max_strainranges: see stress outputs ^
    """
    vm_strainranges=np.array(vm_strainranges)
    ### access tube,receiver for temp results
    panelName   =str(0)
    tubeName    =str(0)
    panel = model.panels[panelName]
    tube = panel.tubes[tubeName]           
    quad = tube.quadrature_results
    ### stresses
    # get each timestep's maximum index
    max_stress_indices=[]
    max_stresses=[]
    temps_at_max_stress=[]
    temps_at_max_strain=[]
    for i in range(vm_stresses.shape[0]): # for loop along the time dimension
        vm_stresses_i=vm_stresses[i,:,:]
        max_index_squashed=np.argmax(vm_stresses_i)
        max_loc           =np.unravel_index(max_index_squashed,vm_stresses_i.shape)
        max_stress_indices.append(max_loc)
        max_stresses.append(vm_stresses_i[max_loc])
        temps_at_max_stress.append(quad['temperature'][i][max_loc])
    ### strains
    max_strainrange_indices=[]
    max_strainranges=[]
    for i in range(vm_strainranges.shape[0]): # for loop along the time dimension
        vm_strainranges_i=vm_strainranges[i,:,:]
        max_index_squashed=np.argmax(vm_strainranges_i)
        max_loc           =np.unravel_index(max_index_squashed,vm_strainranges_i.shape)
        max_strainrange_indices.append(max_loc)
        max_strainranges.append(vm_strainranges_i[max_loc])
        temps_at_max_strain.append(quad['temperature'][i][max_loc])

    return max_stress_indices, max_stresses, temps_at_max_stress, max_strainrange_indices, max_strainranges, temps_at_max_strain

def get_quad_indexed_results_general(model, vm_strainranges, vm_strains, vm_stresses, index):
    """
    purpose: return quadrature results for index (usually crown) of interest.
    ---
    model - objects with results
    vm_strainranges- list (ntimes x elements x 4) pre-computed strain ranges
    vm_strains     - list (ntimes x elements x 4) pre-computed strains
    vm_stresses    - np array (ntimes x elements x 4) pre-computed vm_stresses
    index          - (int) index of interest
    --- all returns are lists ---
    vm_strains_at_loc      - (ntimes x 4) strains for element of interest
    vm_strainranges_at_loc - (ntimes x 4) strainranges for element of interest
    vm_stresses_at_loc     - (ntimes x 4) stresses for element of interest
    temps_at_loc           - (ntimes x 4) temperatures for element of interest
    """
    vm_strainranges=np.array(vm_strainranges)
    vm_strains     =np.array(vm_strains)
    ## access tube,receiver for temp results
    panelName   =str(0)
    tubeName    =str(0)
    panel = model.panels[panelName]
    tube = panel.tubes[tubeName]           
    quad = tube.quadrature_results

    # loop through all timepoints, extract desired data
    vm_strains_at_loc=[]
    vm_strainranges_at_loc=[]
    vm_stresses_at_loc=[]
    temps_at_loc=[]
    for i in range(vm_strainranges.shape[0]): 
        vm_strains_at_loc.append(vm_strains[i,index,:])
        vm_strainranges_at_loc.append(vm_strainranges[i,index,:])
        vm_stresses_at_loc.append(vm_stresses[i,index,:])
        temps_at_loc.append(quad['temperature'][i,index,:])
    return vm_strains_at_loc, vm_strainranges_at_loc, vm_stresses_at_loc, temps_at_loc

def get_quad_indexed_results_selected(model, index, requested_variables):
    """
    purpose: return quadrature results for index (usually crown) of interest.
    ---
    model - objects with results
    index          - (int) index of interest
    requested_variables - keywords of the deisred quadrature results
    --- all returns are lists netimes x 4---
    results_dict - dictionary with keywords 'variables' and logged values
    """
    ## access tube,receiver for temp results
    panelName   =str(0)
    tubeName    =str(0)
    panel = model.panels[panelName]
    tube = panel.tubes[tubeName]           
    quad = tube.quadrature_results
    ##
    results_dict={}
    for variable in requested_variables:
        values_list=[]
        for i in range(quad['temperature'].shape[0]):
            values_list.append(quad[variable][i,index,:])
        results_dict[variable]=values_list

    return results_dict


def get_vm_stuff(receiver):
    """
    pre-computes vm results at each quadrature node for use in other functions
    ---
    vm_strains, vm_strainranges, vm_stresses - (ntimes x elements x 4)
    """
    tube=receiver.panels['0'].tubes['0'] # assuming there's only 1 tube per simulation
    #### vm strains, adapted from the strain range calculation in srlife's damage module: cycle_fatigue method
    # Identify cycle boundaries
    tm = np.mod(tube.times, receiver.period)
    inds = list(np.where(tm == 0)[0])
    if len(inds) != (receiver.days + 1):
        raise ValueError(
            "Tube times not compatible with the receiver"
            " number of days and cycle period!"
        )  
    ## calculate strain per day, which makes the subsequent strain range calculations easier
    # multiply all strains by their respective factors
    nu = 0.5
    strain_names = [
            "mechanical_strain_xx",
            "mechanical_strain_yy",
            "mechanical_strain_zz",
            "mechanical_strain_yz",
            "mechanical_strain_xz",
            "mechanical_strain_xy",
                    ]
    strain_factors = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]

    
    vm_strains     =[]
    vm_strainranges=[]

    for ii in range(receiver.days):
        # if ii < receiver.days-1: # indice set can be open on end for all but last day
        start=inds[ii]
        end  =inds[ii+1] if (ii < receiver.days -1) else None # last day's index needs to include final point
        strains = np.array(
                            [
                                ef * tube.quadrature_results[en][start : end ]
                                for en, ef in zip(strain_names, strain_factors)
                            ]
                        )
        for jj in range(strains.shape[1]): # looping through all the times in one day
            de = strains[:, 0] - strains[:, jj] # calculates strain range based on starting point of whatever day
            strainranges_jj = (
                np.sqrt(2)
                / (2 * (1 + nu))
                * np.sqrt(
                    (de[0] - de[1]) ** 2
                    + (de[1] - de[2]) ** 2
                    + (de[2] - de[0]) ** 2.0
                    + 3.0 / 2.0 * (de[3] ** 2.0 + de[4] ** 2.0 + de[5] ** 2.0)
                )
            )
            strains_jj = strains[:,jj]
            vm_strains_jj = (
                np.sqrt(2)
                / (2 * (1 + nu))
                * np.sqrt(
                    (strains_jj[0] - strains_jj[1]) ** 2
                    + (strains_jj[1] - strains_jj[2]) ** 2
                    + (strains_jj[2] - strains_jj[0]) ** 2.0
                    + 3.0 / 2.0 * (strains_jj[3] ** 2.0 + strains_jj[4] ** 2.0 + strains_jj[5] ** 2.0)
                )
            )
            vm_strainranges.append(strainranges_jj) #equivalent strain ranges wrt start of day
            vm_strains.append(vm_strains_jj)         #equivalent strains     
    ####

    ### vm stresses, code from srlife's damage module in creep_damage method
    vm_stresses = np.sqrt(
            (
                (
                    tube.quadrature_results["stress_xx"]
                    - tube.quadrature_results["stress_yy"]
                )
                ** 2.0
                + (
                    tube.quadrature_results["stress_yy"]
                    - tube.quadrature_results["stress_zz"]
                )
                ** 2.0
                + (
                    tube.quadrature_results["stress_zz"]
                    - tube.quadrature_results["stress_xx"]
                )
                ** 2.0
                + 6.0
                * (
                    tube.quadrature_results["stress_xy"] ** 2.0
                    + tube.quadrature_results["stress_yz"] ** 2.0
                    + tube.quadrature_results["stress_xz"] ** 2.0
                )
            )
            / 2.0
        )
    return vm_strains, vm_strainranges, vm_stresses

# set filepath info
rootdir=os.getcwd()
os.chdir(rootdir)

if len(sys.argv)>1:
    modelName=sys.argv[1]
else:
    print('no file name provided')
    modelName= 'failure!'


# set the material
material = 'A230'
print(f'using material {material}')
# load the receiver
model = receiver.Receiver.load( (rootdir+'/'+modelName+'.hdf5') )

panel = model.panels['0']
tube = panel.tubes['0']

#Cut down on run time for now by making the tube analyses 2D
for panel in model.panels.values(): # have to access the dictionary (?) by specifying "values"
    for tube in panel.tubes.values():   #see above note
        tube.make_2D(tube.dim[2])   #for loop finds desired tube, then eliminates the height dimension?

# Choose the material models
deformation_mat_string = 'elastic_model' #can use 'base' or 'elastic_model' or 'elastic_creep'
print(f'the deformation model is: {deformation_mat_string}')

thermal_mat, deformation_mat, damage_mat = library.load_material(material, "base", deformation_mat_string, "base")
fluid_mat = library.load_thermal_fluid("32MgCl2-68KCl", "base")

# Setup some solver parameters
params = solverparams.ParameterSet()
params['progress_bars'] =True # Print a progress bar to the screen as we solve
params['nthreads'] = 1 # Solve will run in multithreaded mode, set to number of available cores
# set the number of threads to 1 because at 4 it was still only using 1. Probably because only simulating one tube
# params['thermal']['atol'] = 1.0e-4 # During the standby very little happens, lower the atol to accept this result
params['structural']['atol'] = 1.0e-3 # changed 6/26 was 1e-3. tried 1e-4 but didn't really work
# params['system']['atol'] = 1.0e-3
# params['system']['miter'] = 20
params['structural']['miter'] = 25 #...was 50
params['structural']['verbose'] = True
params['structural']['rtol'] = 1.0e-5 # changed 6/26 was 1e-5. tried 1e-6 but didn't really work
params['thermal']['steady'] = True

# Reset the temperatures each night
# solver.add_heuristic() #see email from Messner


# Choose the solvers, i.e. how we are going to solve the thermal,
# single tube, structural system, and damage calculation problems.
# Right now there is only one option for each
# Define the thermal solver to use in solving the heat transfer problem

thermal_solver = thermal.FiniteDifferenceImplicitThermalSolver(params["thermal"]) #use this if not solving thermohydraulic problem

# Define the structural solver to use in solving the individual tube problems
structural_solver = structural.PythonTubeSolver(params["structural"])
# Define the system solver to use in solving the coupled structural system
system_solver = system.SpringSystemSolver(params["system"])
# Damage model to use in calculating life
damage_model = damage.TimeFractionInteractionDamage(params["damage"]) #damage_rainflow.TimeFractionInteractionDamage(params["damage"])

# The solution manager
solver = managers.SolutionManager(model, thermal_solver, thermal_mat, fluid_mat, structural_solver, deformation_mat, damage_mat, system_solver, damage_model, pset = params)


### execute Srlife solver and save the model

# life = solver.solve_life()
# model.save('srlife_models/solved_models/'+material+'_'+modelName+"_solved"+".hdf5") # NOTE: solved model results use significant disk space. don't use this unless it's a small model with low number of timesteps
# print("Best estimate life: %f daily cycles" % life)

### get the solution values
times = model.panels['0'].tubes['0'].times
vm_strains, vm_strainranges, vm_stresses = get_vm_stuff(model)
max_stress_indices, max_stresses, tempsK_at_max_stress, max_strainrange_indices, max_strainranges, tempsK_at_max_strain = get_quad_vm_max_results(model, vm_strainranges,vm_stresses)

## get results for element of interest
# get the model dimensions:
nt =model.panels['0'].tubes['0'].nt
nr =model.panels['0'].tubes['0'].nr
#
index_el_crown=int( ((nr-1)*nt + (nt/4) - nt)-1 ) -1 # NOTE: index shifted 7/23 to test something. this formula is based on my understanding of the element numbering system. I could be wrong, but checks have confirmed this is usually the location of max stress around solar noon for elastic cases (creep will move max stress point)
print(f"collecting results at index: {index_el_crown}")
vm_strains_at_loc, vm_strainranges_at_loc, vm_stresses_at_loc, tempsK_at_loc = get_quad_indexed_results_general(model, vm_strainranges, vm_strains, vm_stresses, index_el_crown)
# get specific results
wish_list=['strain_xy', 'strain_xx']
granted_wish = get_quad_indexed_results_selected(model, index_el_crown, wish_list)

##

### check if the srlife arrays are equal by calling the creep damage function
# damage_model.fatigue_damage(model.panels['0'].tubes['0'],damage_mat, model)
# damage_model.creep_damage(model.panels['0'].tubes['0'],damage_mat, model)
# arr=np.fromstring(crown_vm[10][1:-1], dtype=float, sep=' ') # this line extracts quad data strings and converts back to arrays
# np.allclose(arr, vm[10,611,:],atol=0.01)                    # this line compares two arrays and sees if they're within tolerance

###

### save all data in a timeseries-based csv format
results = {}
results['times']                    =times
results['period']                   =model.period
results['days']                     =model.days
results['max_stress_indices']       =max_stress_indices
results['max_stresses']             =max_stresses
results['tempsK_at_max_stress']     =tempsK_at_max_stress
results['max_strainrange_indices']  =max_strainrange_indices
results['max_strainranges']         =max_strainranges
results['tempsK_at_max_strain']     =tempsK_at_max_strain
results['vm_strains_at_loc']        =vm_strains_at_loc
results['vm_strainranges_at_loc']   =vm_strainranges_at_loc
results['vm_stresses_at_loc']       =vm_stresses_at_loc
results['tempsK_at_loc']            =tempsK_at_loc
results['strain_xy_at_loc']         =granted_wish['strain_xy']
results['strain_xx_at_loc']         =granted_wish['strain_xx']

results_df=pd.DataFrame(dict([(key, pd.Series(value)) for key, value in results.items()])) # prevents need for all columns to be same length. from here: https://www.statology.org/pandas-dataframe-from-dict-with-different-length/
results_df.to_csv((f'{modelName}_with_{material}_results.csv'),index=False)
###