import numpy as np
import matplotlib.pyplot as plt



def buildFracNormal(times,dt_ramp,dt_hold):
    #assume that the receiver ramps immediately (i.e. at zero)
    #assume that the receiver startup and shutdown take the same amount of time
    #BASED ON A 24 HR DAY
    start_shutdown=dt_ramp+dt_hold
    end_shutdown=2*dt_ramp+dt_hold
    ramp_slope=1/dt_ramp    #the forcting function should rampup from 0 to one in a dt_ramp time
    aArray=ramp_slope*times[np.where(times[:]<= dt_ramp)]
    times_in_hold=times[np.where( (times[:]>dt_ramp) & (times[:]<= start_shutdown) )]
    bArray=np.ones(len(times_in_hold))
    times_in_shutdown=times[np.where( (times[:]>start_shutdown) & (times[:]<end_shutdown) )]
    cArray=1-ramp_slope*(times_in_shutdown-start_shutdown)
    times_in_off=times[np.where(times[:]>=end_shutdown)]
    dArray=np.zeros(len(times_in_off))
    activateFrac=np.concatenate([aArray,bArray,cArray,dArray])
    return activateFrac

def buildFrac_NeverOff(times,dt_ramp,dt_hold):
    #assume that the receiver ramps immediately (i.e. at zero)
    #assume that the receiver startup and shutdown take the same amount of time
    #BASED ON A 24 HR DAY
    #basically same thing as buildFracNormal but no off period
    start_shutdown=dt_ramp+dt_hold
    end_shutdown=2*dt_ramp+dt_hold
    ramp_slope=1/dt_ramp    #the forcting function should rampup from 0 to one in a dt_ramp time
    aArray=ramp_slope*times[np.where(times[:]<= dt_ramp)]
    times_in_hold=times[np.where( (times[:]>dt_ramp) & (times[:]<= start_shutdown) )]
    bArray=np.ones(len(times_in_hold))
    times_in_shutdown=times[np.where( (times[:]>start_shutdown)  )]
    cArray=1-ramp_slope*(times_in_shutdown-start_shutdown)
    activateFrac=np.concatenate([aArray,bArray,cArray])
    return activateFrac

def buildFracCos(times,dt_op):
    #assume that the receiver ramps immediately (i.e. at zero)
    #purely cosinusoidal frac for entire dt_op
    #BASED ON A 24 HR DAY
    # start_shutdown=dt_ramp+dt_hold
    # end_shutdown=2*dt_ramp+dt_hold
    aArray=np.sin(np.pi*times[np.where(times[:]<= dt_op)]/dt_op)
    times_off=times[np.where( (times[:]>dt_op) ) ]
    bArray=np.zeros(len(times_off))
    activateFrac=np.concatenate([aArray,bArray])
    return activateFrac

def buildFracCos_wPH_ow(times,t_op,t_ph,f_ph):
    '''
    mimics cosinusoidally varying outer wall crown temperature
    times-- should be the first cycle
    t_op -- the operation duration (hrs)
    t_ph -- duration to reach preheat (hrs)
    f_ph -- T_f/T_peak: the preheat point reached before operation (should be minus Tbase)
    '''
    #assume that the receiver ramps immediately (i.e. at zero)
    #purely cosinusoidal frac for entire dt_op
    #BASED ON A 24 HR DAY
    start_shutdown=t_ph+t_op
    end_shutdown=t_op+2*t_ph

    ramp_slope=f_ph/t_ph   #the forcting function should rampup from 0 to one in a dt_ramp time
    phArray=ramp_slope*times[np.where(times[:]<= t_ph)] # the preheat period

    opArray=(1-f_ph)*np.sin( np.pi*(times[np.where((times[:]> t_ph) & (times[:]<= start_shutdown) )]-t_ph)/t_op )
    opArray=opArray+f_ph


    cdArrayTimes=times[np.where( (times[:]>start_shutdown) & (times[:]<end_shutdown) )]
    cArray=f_ph-ramp_slope*(cdArrayTimes-start_shutdown) # the cooldown period

    times_off=times[np.where( (times[:]>=end_shutdown) ) ]
    offArray=np.zeros(len(times_off))

    activateFrac=np.concatenate([phArray,opArray,cArray,offArray])

    return activateFrac

def buildDeltaTFracCos_wPH(times,t_op,t_ph):
    '''
    makes fraction for delta T progression but with zeros during preheat period
    times-- should be the first cycle
    t_op -- the operation duration (hrs)
    t_ph -- duration to reach preheat (hrs)
    f_ph -- T_f/T_peak: the preheat point reached before operation (should be minus Tbase)
    '''
    #assume that the receiver ramps immediately (i.e. at zero)
    #purely cosinusoidal frac for entire dt_op
    #BASED ON A 24 HR DAY
    start_op=t_ph
    end_op=t_op+t_ph

    times_before_op=times[np.where( (times[:]<=start_op) ) ]
    preOpArray=np.zeros(len(times_before_op))

    opArray=np.sin( np.pi*(times[np.where((times[:]> start_op) & (times[:]<= end_op) )]-t_ph)/t_op )

    times_post_op=times[np.where( (times[:]>end_op) ) ]
    postOpArray=np.zeros(len(times_post_op))

    activateFrac=np.concatenate([preOpArray,opArray,postOpArray])

    return activateFrac

def buildCosFracOuter(times,dt_ramp,dt_collect,startupVal):

    start_shutdown=dt_ramp+dt_collect
    end_shutdown=2*dt_ramp+dt_collect
    
    ramp_slope=startupVal/dt_ramp    #the forcing function should rampup from 0 to one in a dt_ramp time
    aArray=ramp_slope*times[np.where(times[:]<= dt_ramp)]

    times_collecting=times[np.where( (times[:]>dt_ramp) & (times[:]<= start_shutdown) )]
    amp=1-startupVal
    bArray=amp*np.sin( np.pi*(times_collecting-dt_ramp)/dt_collect ) + startupVal

    times_in_shutdown=times[np.where( (times[:]>start_shutdown) & (times[:]<end_shutdown) )]
    cArray=startupVal-ramp_slope*(times_in_shutdown-start_shutdown)

    times_in_off=times[np.where(times[:]>=end_shutdown)]
    dArray=np.zeros(len(times_in_off))
    activateFrac=np.concatenate([aArray,bArray,cArray,dArray])

    return activateFrac

def buildCosFracInner(times,dt_ramp,dt_collect,startupVal,maxVal):
    # dt_ramp: ramp time [hrs]
    # dt_collect: time that the receiver is collecting solar energy. Does NOT include either ramp times
    # startupVal: the temperature fraction (Tpreheat/TouterPeak) that the receiver preheats to before collection begins
    # maxVal: the final temperature fraction (TinnerPeak/TouterPeak) that the inner temperature holds at
    start_shutdown=dt_ramp+dt_collect
    end_shutdown=2*dt_ramp+dt_collect
    
    ramp_slope=startupVal/dt_ramp    #the forcing function should rampup from 0 to one in a dt_ramp time
    aArray=ramp_slope*times[np.where(times[:]<= dt_ramp)]

    times_collecting=times[np.where( (times[:]>dt_ramp) & (times[:]<= start_shutdown) )]
    amp=1-startupVal
    bArray=np.zeros(len(times_collecting))
    for ii in range(len(times_collecting)):
        bArrayProposed=amp*np.sin( np.pi*(times_collecting[ii]-dt_ramp)/dt_collect ) + startupVal
        if bArrayProposed >= maxVal:
            bArray[ii]=maxVal
        else:
            bArray[ii]=bArrayProposed

    times_in_shutdown=times[np.where( (times[:]>start_shutdown) & (times[:]<end_shutdown) )]
    cArray=startupVal-ramp_slope*(times_in_shutdown-start_shutdown)

    times_in_off=times[np.where(times[:]>=end_shutdown)]
    dArray=np.zeros(len(times_in_off))
    activateFrac=np.concatenate([aArray,bArray,cArray,dArray])

    return activateFrac

def innerTemp(tArray, T_in_bot, T_in_top, T_base, nt, nz): #generates an array of dimension: time x theta x z
    #generate Time Variation
    
    T_in_z = np.linspace((T_in_bot-T_base),(T_in_top-T_base), nz)
    
    thetaArray = np.ones(nt)
    innerArray = (tArray[:, None, None]*thetaArray[None,:,None]*(T_in_z[None, None, :]+T_base) ) 
    return innerArray

def buildSurfCosine(valArray,thetaArray,fracArray,deltaT,nt,nz): #assumes no variance in z, but does let temp vary with theta along hotside accoring to thetaArray
    valArray=np.zeros([valArray.size,nt,nz]) + valArray[:,None,None] #expand the cloud induced fractional values to 3 dimensions. this should normally be Tinner values
    thetaArray=thetaArray*deltaT #adds angle variation to deltaT, like a cosine flux
    thetaArray=np.zeros([nt,nz])+thetaArray[:,None] #expand the size of thetaArray to an nt x nz
    thetaArray=thetaArray[None,:,:]*fracArray[:,None,None] 
    returnArray=valArray+thetaArray #add the deltaT to every location
    return returnArray

def buildSurfUniTemp(valArray,thetaArray,nz): #should generate an array of dimension: ntime x nt x nz
#    thetaArray = np.ones(nt) #does not vary in theta #old way. as of 11/15 we are implementing a cosine temperature profile on hotside
   zArray = np.ones(nz) #does not vary in z
   returnArray = (valArray[:,None,None]*thetaArray[None,:,None]*zArray[None,None,:])
   return returnArray

def buildCrownTempFlatHold(times,period,days,NsubSteps,t_ramp,t_hold,T_f,deltaT_total):
    ### function designed to generate single temperature profile of the crown temperature v/s time
    # does not consider off time because this is not used in creep damage analysis
    # inputs :  period (hrs)
    #           days
    #           NsubSteps (1/hrs)
    #           t_ramp & t_hold (hrs)
    #           T_f & deltaT_total (C)
    # outputs: crownTempArray (NsubSteps*period*days+1) x 1 array
    DaySteps=NsubSteps*period #predicts the array length of one day's ntimes
    ## zero everything so that we can add base temp in later
    T_base = 30 # C
    T_f-=T_base
    # create time variance of inner temperature, normalized by the fluid temperature
    fFracArrayInner=buildFracNormal(times[:int(DaySteps+1)],t_ramp,t_hold) #calculate the time varying temperature component for each point in time
    fArrayInner=T_f*fFracArrayInner  #calculate the actual timeseries temperatures above the T_base
    crownTempArrayDay1=fArrayInner + fFracArrayInner*deltaT_total
    crownTempArrayDayN=np.tile(crownTempArrayDay1[1:int(DaySteps+1)],(int(days-1) ))
    crownTempArray=np.concatenate( (crownTempArrayDay1,crownTempArrayDayN) )
    # add back in the T_base
    crownTempArray+=T_base  
    return crownTempArray

def buildCrownTempCosine(times,period,days,NsubSteps,t_ramp,t_hold,T_f,deltaT_total):
    # does not consider off time because this is not used in creep damage analysis
    # inputs :  period (hrs)
    #           days
    #           NsubSteps (1/hrs)
    #           t_ramp & t_hold (hrs)
    #           T_f & deltaT_total (C)
    # outputs: crownTempArray (NsubSteps*period*days+1) x 1 array
    DaySteps=NsubSteps*period #predicts the array length of one day's ntimes
    ## zero everything so that we can add base temp in later
    T_base = 30 # C
    T_f-=T_base
    # create time variance of inner temperature, normalized by the fluid temperature
    dt_hold=t_hold+2*t_ramp
    fFracArrayInner=buildFracCos(times[:int(DaySteps+1)],dt_hold) #calculate the time varying temperature component for each point in time
    fArrayInner=T_f*fFracArrayInner  #calculate the actual timeseries temperatures above the T_base
    crownTempArrayDay1=fArrayInner + fFracArrayInner*deltaT_total
    crownTempArrayDayN=np.tile(crownTempArrayDay1[1:int(DaySteps+1)],(int(days-1) ))
    crownTempArray=np.concatenate( (crownTempArrayDay1,crownTempArrayDayN) )
    # add back in the T_base
    crownTempArray+=T_base  
    return crownTempArray

def buildCrownTemp_wPH(times,period,days,Nsubsteps,t_op,t_ph,Tf,dT):
    '''
    times: all times from simulation (hrs)
    period:duration of each day (hrs)
    days  :number of days in simulation   (days)
    Nsubsteps: unitless number of timesteps per hour
    t_op: the hold time each day (hrs)
    t_ph: the preheat time each day (hrs)
    Tf: the unadjusted fluid temp (C)
    dT: the unadjusted total delta T (C)

    returns: crownTempArray: daily temperature profile at the crown in (C)
    '''
    DaySteps=Nsubsteps*period
    #adjust Tf by Tbase
    T_base=30
    Tf_adj=Tf-T_base
    #build frac for one day:
    crownTempArrayFrac=buildFracCos_wPH_ow(times[:int(DaySteps+1)],t_op,t_ph,(Tf_adj/(Tf_adj+dT)))
    #scale the time varying temp profile:
    crownTempArrayAdj=(Tf_adj+dT)*crownTempArrayFrac
    #adjust by Tbase:
    crownTempArrayDay1=crownTempArrayAdj+T_base
    crownTempArrayDayN=np.tile(crownTempArrayDay1[1:int(DaySteps+1)],(int(days-1) ))
    crownTempArray=np.concatenate( (crownTempArrayDay1,crownTempArrayDayN) )

    return crownTempArray


def buildHoldTracker(times,dt_ramp,dt_hold):
    #this function is 1 if in the hold time, 0 all other places
    #assume that the receiver ramps immediately (i.e. at zero)
    #assume that the receiver startup and shutdown take the same amount of time
    #BASED ON A 24 HR DAY
    start_shutdown=dt_ramp+dt_hold
    times_in_ramp=times[np.where(times[:]<= dt_ramp)]
    aArray=np.zeros(len(times_in_ramp))
    times_in_hold=times[np.where( (times[:]>dt_ramp) & (times[:]<= start_shutdown) )]
    bArray=np.ones(len(times_in_hold))
    times_post_hold=times[np.where( (times[:]>start_shutdown) )]
    cArray=np.zeros(len(times_post_hold))
    FracArray=np.concatenate([aArray,bArray,cArray])
    return FracArray

def pressure(pmax,times,t_rampP,t_shutdownP): #generates an array of dimension: time
    '''
    pmax (MPa)
    times (hrs)
    t_rampP (hrs) the time it takes to ramp up pressure
    t_shutdownP (hrs) the timestamp when ramp down STARTS
    '''
    rampup= (pmax/t_rampP)*times[np.where(times[:] < t_rampP)] #ramp up
    op = pmax*np.ones(len(np.where((times[:] <= t_shutdownP) & (times[:] >= t_rampP))[0]))
    rampdown = pmax* (1+(-1/(t_rampP))*(times[np.where((times[:] > t_shutdownP) & (times[:] < (t_shutdownP + t_rampP) ))]- t_shutdownP) )
    night = 0*np.ones(len(np.where( times[:] >= (t_shutdownP+t_rampP) )[0]))
    pArray = np.concatenate([rampup,op,rampdown,night], axis=None)
    return pArray

def makeFlux(fArray_mesh,theta_mesh,z_mesh,h_max):
    flux = fArray_mesh * np.cos(theta_mesh) * h_max 
    return flux


def makeThetaArray(pts): #varies angle between -pi/2 and pi/2. Needs input to be a negative
    A=np.linspace(-np.pi/2,np.pi/2,pts) #varies theta between -pi/2 and pi/2
    # B=np.ones(int(pts/2))*(np.pi/2) #causes flux to be zero
    thetaArray = np.cos(A) #np.concatenate((A,B))
    return thetaArray #returns fractions of 1

if __name__ == "__main__": #main is built for testing functions
    period = 24.0 # Loading cycle period, hours ... was 24.0
    days = 6 # Number of cycles represented in the problem ... was 1
    substeps=40
    DaySteps=substeps*period #predicts the array length of one day's ntimes
    times = np.linspace(0,period*days,int(period*days)*substeps+1)
    dt_ramp=0.5
    t_hold=12
    t_ph=0.5
    f_ph=0.2
    # activateFrac=buildFracCos_wPH_ow(times[:int(DaySteps+1)],t_hold,t_ph,f_ph)
    activateFrac=buildDeltaTFracCos_wPH(times[:int(DaySteps+1)],t_hold,t_ph)
    print("shape of the activate frac array is: %0.0f" % activateFrac.shape)

    plt.plot(times[:int(DaySteps+1)],activateFrac)
    plt.show()

    # Tpeak=600
    # Tbase=50
    # valArray=(Tpeak-Tbase)*activateFrac
    # valArray=valArray+50
    # print(valArray)
    # plt.plot(times[:int(DaySteps+1)],valArray)
    # plt.show()

    Touter=buildCosFracOuter(times[:int(DaySteps+1)],dt_ramp,t_hold,0.4)
    Tinner=buildCosFracInner(times[:int(DaySteps+1)],dt_ramp,t_hold,0.4,0.75)
    plt.plot(times[:int(DaySteps+1)],Touter)
    plt.plot(times[:int(DaySteps+1)],Tinner)
    plt.show()