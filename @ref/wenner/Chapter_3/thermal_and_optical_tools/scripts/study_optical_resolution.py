"""
created by jwenner on 10/19/2025 to study effect of optical resolution on incident power
"""
import sys
sys.path.append('..') # thermal model is in upper level folder
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
import util

## instantiate a full thermal model
model = steady_state_analysis_jwenn.SteadyStateAnalysis()
#--- Read existing receiver design from file and/or update receiver design input parameters
inputName = 'resolution_study_receiver'
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
model.analysis_mode = 'design_point' #'design_day'  ,- was using this 1/27/25, og comment -># 'design_point', 'design_day', 'three_day' (summer solstice, equinox, winter solstice), 'selected_days', 'user_defined' # was using "selected_days"
model.analysis_days = [172] # can manually set this but analysis_mode will default set this as well
substeps=0.5                    # number of model substeps per hour
model.delta_hour = 1/substeps                # Time spacing between simulated time points (hr)
model.is_half_day = False           # Simulate only half of the day?
model.dni_setting = 'user-defined-noon'  
model.user_defined_dni = {172:950, 264:980, 355:930} # User-defined DNI at solar noon on each simulated day.  Used in conjunction with clearsky-DNI to set DNI per time point
model.ambient_setting = 'user-defined'
model.user_defined_Tamb = 25+273.15    # Constant user-defined ambient T (K)
model.user_defined_RH = 25             # Constant user-defined relative humidity
model.user_defined_vwind10 = 0         # Constant user-defined wind speed (m/s at 10m height)

## simulate flux profile
model.initialize()
doy         =172
hour_of_day =11.819
model.operating_conditions.day = doy
model.operating_conditions.tod = hour_of_day
hour_offset = util.get_offset(model.receiver.site, doy, hour_of_day)
model.operating_conditions.hour_offset = hour_offset
model.set_weather()
inc_flux, opteff = model.SF.get_flux_profile_wCP(doy, hour_of_day = hour_of_day)

## plot the flux profile
fontsize = 14
fluxFig, fluxAx = plt.subplots()
im = fluxAx.imshow(inc_flux, extent=[-model.receiver.D/2, model.receiver.D/2, 0, model.receiver.H])
cbr = fluxFig.colorbar(im)
cbr.set_label(label='incident flux [kW/m$^2$]', size=fontsize)
cbr.ax.tick_params(labelsize=fontsize)
fluxAx.set_xlabel('x coordinate (m)',size=fontsize)
fluxAx.set_ylabel('y coordinate (m)',size=fontsize)

# plt.savefig('imgs/latest_flux_profile.jpg', dpi=300)
plt.show()
plt.close()

## calculate total incident power and maximum flux
Q_inc_rec =np.average(inc_flux)*model.receiver.D*model.receiver.H
print(f'incident power is {Q_inc_rec:.2f}')
q_flux_max=np.max(inc_flux)
print(f'maximum flux is {q_flux_max:.2f}')

## save flux profile
savename =f'aiming/flux_array_Qdes{model.receiver.Qdes:.0f}_res{model.receiver.n_flux_x}.json'
helpers_thermal_model.np_to_json(['inc_fluxes'], [np.array(inc_flux)], savename)