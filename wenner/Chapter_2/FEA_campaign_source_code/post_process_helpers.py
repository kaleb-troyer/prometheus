"""
created by jwenner on 6/15/25
includes functions actually used for post-processing
"""

import numpy as np
import scipy.signal as sPsig
import matplotlib.pyplot as plt

def get_daily_strainranges_at_loc(results_df,ndays_select):
    """
    -extracts maximum daily strainranges from dataframe at the queried location during simulation
    -grabs the maximum result from the quadrature result
    ---
    returns array with size ndays
    """
    # extract the values and reorganize into an array
    values=results_df['vm_strainranges_at_loc'].values
    values=[np.max(np.fromstring(element[1:-1], dtype=float, sep=' ')) for element in values] # this line extracts quad data strings and converts back to arrays
    values=np.array(values)
    # find the daily max
    times       =results_df['times'].values
    period      =results_df['period'].values[0]
    days        =results_df['days'].values[0]
    nSubsteps   =times.size // (period*days)

    max_indices=get_peak_indices(values, days, period, nSubsteps)
    strainranges=values[max_indices][:ndays_select]  # only selects the number of days we are interested in 
    # strainranges=np.array([values[max_ind] for max_ind in max_indices])

    return strainranges


def get_daily_creep_damages_at_loc(results_df, damage_mat, material, ndays_select):
    '''
    calculates creep damage using crown stress, times, and crown temperature
    results_df - should contain stress, times, and temperature (K)
    damage_mat - damage model
    ---
    returns array of daily creep damages
    '''
    sigmas  =results_df['vm_stresses_at_loc'].values
    sigmas  =[np.max(np.fromstring(element[1:-1], dtype=float, sep=' ')) for element in sigmas] # this line extracts quad data strings and converts back to arrays
    sigmas  =np.array(sigmas)

    temps_K =results_df['tempsK_at_loc'].values
    temps_K =[np.max(np.fromstring(element[1:-1], dtype=float, sep=' ')) for element in temps_K] # this line extracts quad data strings and converts back to arrays
    temps_K =np.array(temps_K)

    times   =results_df['times'].values
    period  =results_df['period'].values[0]
    days    =int(results_df['days'].values[0])   

    if material == 'A230':  #using Eno correlation for A230 because SRLIFE database is admittedly sparse for this material. 
        dcs = calc_creep_damage_laporte(results_df,sigmas,temps_K,times,period,days)
    else:
        tR = damage_mat.time_to_rupture("averageRupture", temps_K, sigmas)
        dts = np.diff(times)
        time_dmg = dts / tR[1:]
        inds=id_cycles(results_df)
        dcs = np.array(
                [
                    np.sum(time_dmg[inds[i] : inds[i + 1]], axis=0)
                    for i in range(days)
                ]
            )

    return dcs[:ndays_select]        # only selects the number of days we are interested in 


def calc_creep_damage_laporte(results_df, sigmas, T_Ks, times, period, days):
    '''
    Correlation is from Laporte et al. (2021) "Material selection for solar central receiver tubes", adapted from Eno et al. source
    T_Ks    - (K)
    sigma   - (MPa)
    tR      - (hrs)
    '''
    ### get tR from MDM in LaporteMaterials2021
    B0=-26.27
    B1= 44158
    B2=4.72
    B3=-11337
    sigmas=sigmas+0.00001 # adding a small nonzero number to remove log10 error on first data point
    tRs=10**(B0+(B1/T_Ks)+np.log10(sigmas**B2)+np.log10(sigmas**(B3/T_Ks)))
    dts = np.diff(times)
    time_dmg = dts / tRs[1:]
    inds=id_cycles(results_df)
    dcs = np.array(
            [
                np.sum(time_dmg[inds[i] : inds[i + 1]], axis=0)
                for i in range(days)
            ]
        )
    return dcs

def id_cycles(results_df):
        """
        Helper to separate out individual cycles by index. Copied from srlife code. Can be found in srlife's damage.py module

        Parameters:
          tube        single tube with results
          receiver    receiver, for metadata
        """
        times   =results_df['times'].values
        period  =results_df['period'].values[0]
        days    =results_df['days'].values[0]

        tm = np.mod(times, period)
        inds = list(np.where(tm == 0)[0])
        if len(inds) != (days + 1):
            raise ValueError(
                "Tube times not compatible with the receiver"
                " number of days and cycle period!"
            )

        return inds

def extrap_20th_day_creep_damage(dcs):
    """
    dcs - array of FEA simulated creep damages
    ----
    returns 20th day creep damage via power law extrapolation of days 3 and 4 
    """
    dc20=extrap_day_w_power_law(np.array([3,4]),dcs[2:4],1)(20)
    
    return dc20

def extrap_day_w_power_law(Narray,Darray,order):
    '''
    returns a lambda function that will predict a single day damage at day N
    Narray: array of cycle points to supply polyfit (integer)
    Darray: array of damage at corresponding points (dimensionless decimals)
    order: polynomial order. Just 1 for now
    '''
    p = np.polyfit(np.log10(Narray), np.log10(Darray), order)
    if order != 1:
        print('only linear log extrapolations. Higher order extrapolations not yet implemented')
    return lambda N, p=p: N**(p[0])* ( 10**(p[1]) )#np.polyval(p, N)

def get_peak_indices(timeseries_data, days, period, nSubsteps, tOff=1):
    """
    returns the indices at which maximum occurs
    timeseries_data     - (varies) a results array from SRLIFE simulation
    days                - (-) number of simulated days
    period              - (hrs) total number of hours in a day/cycle
    nSubsteps           - (-/hr) number of timesteps per hour
    tOff                - (hrs) the cutoff distance between minimas
    """
   
    
    min_inds = sPsig.find_peaks( timeseries_data*(-1), distance=tOff*nSubsteps, width=tOff*nSubsteps )[0]
    min_inds = np.concatenate((np.array([0]),min_inds,np.array([timeseries_data.size-1]))) # find peaks doesn't identify beginning or ending, so must add manually
    
    opMax_inds = np.array([])
    for day in range(int(days) ):
        left=day*period*nSubsteps
        right=left+period*nSubsteps+1
        min_set = min_inds[np.where( (min_inds[:] >= left) & (min_inds[:] <= right) )[0]]
        opMax_ind=np.argmax(timeseries_data[min_set[0]:min_set[1]])+min_set[0]
        opMax_inds=np.concatenate( (opMax_inds,np.array([opMax_ind])) )
    opMax_inds=opMax_inds.astype(int)

    ## plotting the min and max points as a check
    # fig,ax = plt.subplots()
    # ax.plot(timeseries_data,linewidth=1)
    # ax.scatter(min_inds,timeseries_data[min_inds],s=5,color='r',label='minimum points')
    # ax.scatter(opMax_inds,timeseries_data[opMax_inds],s=5, color='k',label='maximum points')
    # plt.show()
    ##
    return opMax_inds

def cycles_to_fail_lawless(damage_mat, temp, erange,cutoff_bool):
        """
        Returns fatigue cycles to failure at a given temperature and strain range.
        Uses the same curves even if the strain range is below the cutoff  
        Parameters:
          pname:       property name ("nominalFatigue")
          erange:      strain range in mm/mm
          temp:        temperature in (C)
        """
        pname = "nominalFatigue"
        pdata = damage_mat.data[pname]
        temp_K=temp+273.15
        T, a, n, cutoff = [], [], [], []
        for i in pdata:
            T.append(destring_array(pdata[i]["T"]))
            a.append(destring_array(pdata[i]["a"]))
            n.append(destring_array(pdata[i]["n"]))
            cutoff.append(destring_array(pdata[i]["cutoff"]))

            if np.array(a).shape != np.array(n).shape:
                raise ValueError("\tThe lists of a and n must have equal lengths!")

        inds = np.array(T).argsort(axis=0)
        T = np.array(T)[inds]
        a = np.array(a)[inds]
        n = np.array(n)[inds]
        cutoff = np.array(cutoff)[inds]
        if temp_K > max(T):
            raise ValueError(
                "\ttemperature is out of range for cycle to failure determination"
            )
        for i in range(np.size(T, axis=0)):
            if temp_K <= T[i]:
                polysum = 0.0
                if erange <= cutoff[i] and cutoff_bool==True:
                    erange = cutoff[i][0][0]
                elif erange<=cutoff[i] and cutoff_bool==False:
                    print('extrapolating curve')
                for b, m in zip(a[i][0], n[i][0]):
                    polysum += b * np.log10(erange) ** m
                break

        return 10**polysum

def destring_array(string):
    """
    Make an array from a space separated string
    I took this straight from SRLIFE
    """
    return np.array(list(map(float, string.split(" "))))