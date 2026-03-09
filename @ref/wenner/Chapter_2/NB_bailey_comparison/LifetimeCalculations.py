# this file contains necessary functions for calculating damage. These functions are
# found in the former damage.py file and plot_cosineCases_p5.py file
import numpy as np


#! to be used to calculate instantaneous damage a function of stress and temp
def instantaneous_dmg(sigmas,Ts,times):
    # Convert inputs to numpy arrays if they aren't already
    sigmas = np.array(sigmas)
    Ts = np.array(Ts)
    T_Ks=Ts+273.15
    ### get tR from MDM in LaporteMaterials2021
    B0=-26.27
    B1= 44158
    B2=4.72
    B3=-11337
    tRs=10**(B0+(B1/T_Ks)+np.log10(sigmas**B2)+np.log10(sigmas**(B3/T_Ks)))
    dts = np.diff(times)
    time_dmg = dts / tRs[1:]
    return sigmas, Ts, time_dmg

#! Following 2 functions used to calculate creep damage from FEA files (used to be in damage.py)
def id_cycles(times,period):
    """
    Helper to separate out individual cycles by index
    Aadapted from SRLIFE
    Parameters:
        times (hrs)
        period (hrs)
    """
    tm = np.mod(times, period)
    inds = list(np.where(tm == 0)[0])
    # if len(inds) != (receiver.days + 1):
    #     raise ValueError(
    #         "Tube times not compatible with the receiver"
    #         " number of days and cycle period!"
    #     )

    return inds

def calcCreep_laporte(sigmas,Ts,times,period,days):
    '''
    T - (C)
    sigma - (MPa)
    tR - (hrs)
    Returns the damage for each day
    '''
    
    # Convert inputs to numpy arrays if they aren't already
    sigmas = np.array(sigmas)
    Ts = np.array(Ts)
    T_Ks=Ts+273.15
    ### get tR from MDM in LaporteMaterials2021
    B0=-26.27
    B1= 44158
    B2=4.72
    B3=-11337
    tRs=10**(B0+(B1/T_Ks)+np.log10(sigmas**B2)+np.log10(sigmas**(B3/T_Ks)))
    dts = np.diff(times)
    time_dmg = dts / tRs[1:]
    inds=id_cycles(times,period)
    dcs = np.array(
            [
                np.sum(time_dmg[inds[i] : inds[i + 1]], axis=0)
                for i in range(days)
            ]
        )
    return dcs

