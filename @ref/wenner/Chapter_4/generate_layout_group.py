import sys
sys.path.append('../Chapter_3/thermal_and_optical_tools/') # thermal model is in upper level folder
sys.path.append('../Chapter_3/damage_tool/') # damage tool
sys.path.append('./cost_model/')
import os, csv
import matplotlib.pyplot as plt
import numpy as np
import random
import json

"""
adapted on 10/2/25 by jwenner

used to create a series of CoPylot layouts for aspect ratio study
"""

from copylot import CoPylot

eta_r = 0.91 # estimated overall receiver efficiency
cp = CoPylot() # create copylot instance
r  = cp.data_create() # specific instance case. R is just a memory ID
cp.api_callback_create(r) # i'm not sure if i need this or not
cp.data_set_string(
            r,
            "ambient.0.weather_file",
            "USA CA Daggett (TMY2).csv")

## layout setup info
tht=170
cp.data_set_number(r, "solarfield.0.tht", tht)            # (m)
# cp.data_set_number(r, "solarfield.0.q_des", Qdes)       # (MWt)

## receiver info
cp.data_set_string(r, "receiver.0.rec_type", "Flat plate")

## set range of aspect ratios
A_total  =15*18
print(f'AR determined with a constant area of {A_total} m2')
W_panel  =1
print(f'assuming panel width of {W_panel} m')
eta_opts=[] # overall optical efficiency
eta_ints=[] # image intercept efficiency
### calculate widths from aspect ratio range
# ARs     =np.arange(0.2, 1, 0.1)
# ARs_high=np.arange(1,2.8,0.3)
# ARs     =np.concatenate((ARs, ARs_high))
# ARs     =np.array([0.4, 1, 2.5])
# Hs      =np.sqrt(A_total*ARs)
# Ws      =A_total/Hs
# # calculate the number of panels
# Npanels =np.round(Ws/W_panel)
# # calculate usable widths based on number of panels
# Ws      =Npanels*W_panel
# # enforce only even receiver widths
# Ws      =2*(Ws//2)
### 

Ws      =np.arange(10,36,2)      # receiver width range is set based on previous calculations

# recalculate the Hs now
Hs      =A_total/Ws
# calculate the true aspect ratios
ARs     =Hs/Ws
Qdes    =220    # MWth power to HTF
## loop through ARs
for i, AR in enumerate(ARs):
    H=Hs[i]
    W=Ws[i]
    cp.data_set_number(r, "receiver.0.rec_height", H )       # (m)
    cp.data_set_number(r, "receiver.0.rec_width", W )       # (m)

    # set the maximum field radius as a function of tower height
    cp.data_set_number(r, "land.0.max_scaled_rad", 13)
    cp.data_set_number(r, "solarfield.0.is_opt_zoning", 0)

    # ## get the layout
    # cp.generate_layout(r)
    # field = cp.get_layout_info(r)


    ## check power output on design day: 172 at noon
    cp.data_set_number(r,"fluxsim.0.flux_month",6)
    cp.data_set_number(r,"fluxsim.0.flux_day",21)
    cp.data_set_number(r,"fluxsim.0.flux_hour",12)

## simulate, iterate to get Qact to equal what we want
# Qrange=range(240,260,10)
# for Qdes in Qrange:
    # Qdes = 200 # (MWth)
    tol = 1 # (MWth)
    Qlow = Qdes-tol
    Qhigh= Qdes+tol
    Qact = 1 # something low to start the loop
    Qdes_nom = Qdes # for starters, the nominal = intended
    ok=False
    step=10
    Qact_old=1
    while ok==False:
        cp.data_set_number(r, "solarfield.0.q_des", Qdes_nom)       # (MWt)

        ## get the layout
        cp.generate_layout(r)
        field = cp.get_layout_info(r)

        cp.simulate(r)
        res_summ = cp.summary_results(r,save_dict=True)
        Qinc = res_summ['Power incident on field']*res_summ['Solar field optical efficiency']/100/1e3 # factors convert efficiency to a decimal and kW to MW
        Qact = Qinc*eta_r
        if abs(Qdes-Qact) < tol: # case that we hit our goal exactly
            ok=True
        elif (Qact > Qdes+tol) and (Qact_old > Qdes+tol): # case that we are still high after one decrement. We don't want to decrease the step size or it'll take too long
            Qdes_nom = Qdes_nom-step    
        elif (Qact > Qdes+tol) and (Qact_old < Qdes-tol): # case that our guess exceeded goal, then we should decrease the step count and try again
            step=step/2
            Qdes_nom = Qdes_nom-step
        else: # case that we are still low and just need to keep incrementing at current rate
            Qdes_nom = Qdes_nom+step # slowly increase Qdes until we get a large enough heliostat field
        Qact_old = Qact #update the history variable
    txt = "   design Qdot:{input1:.2f}, actual Qdot:{input2:.2f}"
    print(txt.format(input1=Qdes,input2=Qact))

    eta_opt =res_summ['Solar field optical efficiency']
    eta_int =res_summ['Image intercept efficiency']

    print(f'   optical efficiency: {res_summ['Solar field optical efficiency']}')
    ## save the layout after while loop is exited
    field = cp.get_layout_info(r)
    field['zeros']=np.zeros(len(field['x_location'])) # add another column of seemingly pointless zeros
    field_red_info = field.loc[:,['zeros','x_location','y_location','z_location']]
    layoutName = f"layouts/generated_layout_{Qdes:.0f}MWth_{AR:.2f}AR_{W:.2f}width{H:.2f}height_{tht}tht.csv"
    field_red_info.to_csv(layoutName,header=None,index=None)
    ##

    # log the image intercept efficiency and optical effiency for later use
    eta_opts.append(eta_opt)
    eta_ints.append(eta_int)

    # Plotting (default) solar field and flux map
    # Solar Field
    fontsize=16
    field = cp.get_layout_info(r)
    plt.scatter(field['x_location'], field['y_location'], s=1.5)
    plt.xlabel('x coordinate (m)',fontsize=fontsize)
    plt.ylabel('y coordinate (m)', fontsize=fontsize)
    # plt.xlim([-1500,1500])
    plt.xlim([-1050,1050])
    plt.ylim([-50,1950])
    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()
    # picName = "layouts/generated_layout_{Qdes:.0f}MWth_{W}width{H}height_{tht}tht.png".format(Qdes=Qdes,W=W,H=H,tht=tht)
    picName = f"layouts/generated_layout_{Qdes:.0f}MWth_{AR:.2f}AR_{W:.2f}width{H:.2f}height_{tht}tht.png"
    plt.savefig(picName, dpi=300)
    # plt.show()
    plt.close()

# save the efficiency tables
opt_dict={'eta_opt':eta_opts, 'eta_intercepts':eta_ints, 'ARs':ARs.tolist()}
with open('layouts/efficiencies.json', "w") as f:
    json.dump(opt_dict, f)

## plot the intercept efficiency as a function of aspect ratio
fig,ax =plt.subplots()
ax.plot(ARs, eta_ints, linestyle='--', marker='.')
ax.set_xlabel('aspect ratio (H/W)')
ax.set_ylabel('intercept efficiency (-)')
plt.savefig('imgs/optical_efficiency_trend',dpi=300)
plt.show()

# # Plotting (default) solar field and flux map
# # Solar Field
# field = cp.get_layout_info(r)
# plt.scatter(field['x_location'], field['y_location'], s=1.5)
# plt.tight_layout()
# plt.show()

cp.data_free(r)

### useful commands
# cp.data_set_number(r, "fluxsim.0.sigma_limit_x", 1.93)      # (m)
# cp.data_set_number(r, "fluxsim.0.sigma_limit_y", 1.76)      # (m)
# cp.data_set_number(r, "fluxsim.0.flux_day", 20)
# cp.data_set_number(r, "fluxsim.0.flux_month", 3)
# cp.data_set_number(r,"receiver.0.peak_flux",1000) #(kW/m^2)
# cp.data_set_string(r, "receiver.0.flux_profile_type", "User")
# cp.data_set_matrix_from_csv(r, "receiver.0.user_flux_profile", "aiming_scheme.csv" )
# cp.data_get_number(r,"fluxsim.0.flux_solar_el")
# cp.data_get_number(r,"fluxsim.0.flux_solar_az")
# print(f"  design Qdot:{input1:.2f}, actual Qdot:{input2:.2f}")