import os
import pandas as pd
import scipy.interpolate as spInt
import scipy.signal as sPsig
import numpy as np
import matplotlib.pyplot as plt

def getStressDF(Tf,dT):
    ## finds stress DF based on fluid temperature and maximum deltaT
    # only works for A230 since the file name is basically hardcoded...
    # returns: pandas dataframe and the number of substeps SRLIFE used
    exists=True #tells executing code if the DF actually exists
    fileString1 = 'dataframes/p1_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_60substeps_2days_14period_.csv'
    fileString2 = 'dataframes/p1_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_70substeps_2days_14period_.csv'
    fileString3 = 'dataframes/p1_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_80substeps_2days_14period_.csv'
    fileString4 = 'dataframes/p1_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_50substeps_2days_14period_.csv'
    if os.path.exists(fileString1):
        stressDF=pd.read_csv(fileString1)
        NsubSteps=60
    elif os.path.exists(fileString2):
        stressDF=pd.read_csv(fileString2)
        NsubSteps=70
    elif os.path.exists(fileString3):
        stressDF=pd.read_csv(fileString3)
        NsubSteps=80
    elif os.path.exists(fileString4):
        stressDF=pd.read_csv(fileString4)
        NsubSteps=50
    else:
        print('skipped', Tf, dT)
        exists=False    
        stressDF=0
        NsubSteps=0
    return stressDF, NsubSteps, exists

def getStressDF_P4(Tf,dT):
    fileString =  'dataframes/p4_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_30substeps_4days_14period_cosTimeFrac.csv'
    fileString2 = 'dataframes/p4_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_40substeps_4days_14period_cosTimeFrac.csv'
    fileString3 = 'dataframes/p4_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_25substeps_4days_14period_cosTimeFrac.csv'
    fileString4 = 'dataframes/p4_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_15substeps_4days_14period_cosTimeFrac.csv'
    fileString5 = 'dataframes/p4_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_20substeps_4days_14period_cosTimeFrac.csv'

    exists=True
    if os.path.exists(fileString):
        fileData=pd.read_csv(fileString)
        NsubSteps=30
    elif os.path.exists(fileString2):
        fileData=pd.read_csv(fileString2)
        NsubSteps=40
    elif os.path.exists(fileString3):
        fileData=pd.read_csv(fileString3)
        NsubSteps=25
    elif os.path.exists(fileString4):
        fileData=pd.read_csv(fileString4)
        NsubSteps=15
    elif os.path.exists(fileString5):
        fileData=pd.read_csv(fileString5)
        NsubSteps=20
    # elif os.path.exists(fileString4):
    #     fileData=pd.read_csv(fileString4)
    #     NsubSteps=50
    else:
        print('skipped', Tf, dT)
        exists=False
        return 0,0,exists
        
        
    fileData.columns=['index','sigma_crown','sigma_max']                              #this is the csv's dataframe
    daySel=3
    period=14
    # ind=int( (daySel-1)*period*NsubSteps+(period*NsubSteps/2) )
    sigma_crown=fileData.sigma_crown.values #...[fileData.index==ind]
    sigma_max=fileData.sigma_max.values #...[fileData.index==ind] #get the maximum stress on the 3rd day
    # print(strainRange1,strainRange2)
    stressData=pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)  #just need the stresses to be formatted in a way I can add to master DF
    stressData['sigma_crown']=sigma_crown
    stressData['sigma_max']=sigma_max
    stressData['Tf']=Tf
    stressData['dT']=dT
    return stressData,NsubSteps,exists

def getStressDF_P5(Tf,dT):
    fileString =  'dataframes/p5_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_30substeps_4days_14period_cosTimeFrac.csv'
    fileString2 = 'dataframes/p5_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_20substeps_4days_14period_cosTimeFrac.csv'
    fileString3 = 'dataframes/p5_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_30substeps_4days_14period_p5.csv'
    fileString4 = 'dataframes/p5_results/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_20substeps_4days_14period_p5.csv'


    exists=True
    if os.path.exists(fileString):
        fileData=pd.read_csv(fileString)
        NsubSteps=30
    elif os.path.exists(fileString2):
        fileData=pd.read_csv(fileString2)
        NsubSteps=20
    elif os.path.exists(fileString3):
        fileData=pd.read_csv(fileString3)
        NsubSteps=30
    elif os.path.exists(fileString4):
        fileData=pd.read_csv(fileString4)
        NsubSteps=20 

    # elif os.path.exists(fileString4):
    #     fileData=pd.read_csv(fileString4)
    #     NsubSteps=50
    else:
        print('skipped', Tf, dT)
        exists=False
        return 0,0,exists
        
        
    fileData.columns=['index','sigma_crown','sigma_max']                              #this is the csv's dataframe
    daySel=3
    period=14
    # ind=int( (daySel-1)*period*NsubSteps+(period*NsubSteps/2) )
    sigma_crown=fileData.sigma_crown.values #...[fileData.index==ind]
    sigma_max=fileData.sigma_max.values #...[fileData.index==ind] #get the maximum stress on the 3rd day
    # print(strainRange1,strainRange2)
    stressData=pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)  #just need the stresses to be formatted in a way I can add to master DF
    stressData['sigma_crown']=sigma_crown
    stressData['sigma_max']=sigma_max
    stressData['Tf']=Tf
    stressData['dT']=dT
    return stressData,NsubSteps,exists

def getProfilesDF_P5el(Tf,dT):
    fileString =  'dataframes/p5_results/stresses&strains_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_30substeps_4days_14period_p5el.csv'

    exists=True
    if os.path.exists(fileString):
        fileData=pd.read_csv(fileString)
        NsubSteps=30

    # elif os.path.exists(fileString4):
    #     fileData=pd.read_csv(fileString4)
    #     NsubSteps=50
    else:
        print('skipped', Tf, dT)
        exists=False
        return 0,0,exists
        
        
    fileData.columns=['index','sigma_crown','sigma_max','strains']                              #this is the csv's dataframe

    return fileData,NsubSteps,exists

def getStrainRange_P5el(Tf,dT):
    '''
    returns maximum strain range
    '''
    fileString = 'dataframes/p5_results/metrics_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_30substeps_4days_14period_p5el.csv'

    exists=True
    if os.path.exists(fileString):
        fileData=pd.read_csv(fileString)
        NsubSteps=30

    else:
        print('skipped', Tf, dT)
        exists=False
        return 0,0,exists
    fileData=pd.read_csv(fileString)
    fileData.columns=['name','val']   
        

    strainRange3=fileData.val[fileData.name=='strainRangePtMax1']


    return strainRange3,exists


def getStressDF_P7(Tf,dT):

    fileString =  'dataframes/p7_results/stresses_dframe_316H_'+str(Tf)+'Tf_'+str(dT)+'dT_0.4257R_30substeps_4days_14period_p7.csv'
    fileString2 = 'dataframes/p7_results/stresses_dframe_316H_'+str(Tf)+'Tf_'+str(dT)+'dT_0.4257R_40substeps_4days_14period_p7.csv'
    fileString5 = 'dataframes/p7_results/stresses_dframe_316H_'+str(Tf)+'Tf_'+str(dT)+'dT_0.4257R_20substeps_4days_14period_p7.csv'

    exists=True
    if os.path.exists(fileString):
        fileData=pd.read_csv(fileString)
        NsubSteps=30
    elif os.path.exists(fileString2):
        fileData=pd.read_csv(fileString2)
        NsubSteps=40
    elif os.path.exists(fileString5):
        fileData=pd.read_csv(fileString5)
        NsubSteps=20
    else:
        print('skipped', Tf, dT)
        exists=False
        return 0,0,exists
        
        
    fileData.columns=['index','sigma_crown','sigma_max']                              #this is the csv's dataframe

    sigma_crown=fileData.sigma_crown.values #...[fileData.index==ind]
    sigma_max=fileData.sigma_max.values #...[fileData.index==ind] #get the maximum stress on the 3rd day

    stressData=pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)  #just need the stresses to be formatted in a way I can add to master DF
    stressData['sigma_crown']=sigma_crown
    stressData['sigma_max']=sigma_max
    stressData['Tf']=Tf
    stressData['dT']=dT
    
    return stressData,NsubSteps,exists

def getStressDF_P8(Tf,dT):

    fileString  =  'dataframes/p8_results/stresses_dframe_740H_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5265R_30substeps_4days_14period_p8.csv'
    fileString2 =  'dataframes/p8_results/stresses_dframe_740H_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5265R_20substeps_4days_14period_p8.csv'

    exists=True
    if os.path.exists(fileString):
        fileData=pd.read_csv(fileString)
        NsubSteps=30
    elif os.path.exists(fileString2):
        fileData=pd.read_csv(fileString2)
        NsubSteps=20
    else:
        print('skipped', Tf, dT)
        exists=False
        return 0,0,exists
                
    fileData.columns=['index','sigma_crown','sigma_max']                              #this is the csv's dataframe

    sigma_crown=fileData.sigma_crown.values #...[fileData.index==ind]
    sigma_max=fileData.sigma_max.values #...[fileData.index==ind] #get the maximum stress on the 3rd day

    stressData=pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)  #just need the stresses to be formatted in a way I can add to master DF
    stressData['sigma_crown']=sigma_crown
    stressData['sigma_max']=sigma_max
    stressData['Tf']=Tf
    stressData['dT']=dT
    
    return stressData,NsubSteps,exists

def getStressDF_P10(Tf,dT):

    fileString  =  'dataframes/p10_results/stresses_dframe_A282_'+str(Tf)+'Tf_'+str(dT)+'dT_0.4447R_30substeps_4days_14period_p10.csv'
    fileString2 =  'dataframes/p10_results/stresses_dframe_A282_'+str(Tf)+'Tf_'+str(dT)+'dT_0.4447R_20substeps_4days_14period_p10.csv'

    exists=True
    if os.path.exists(fileString):
        fileData=pd.read_csv(fileString)
        NsubSteps=30
    elif os.path.exists(fileString2):
        fileData=pd.read_csv(fileString2)
        NsubSteps=20
    else:
        print('skipped', Tf, dT)
        exists=False
        return 0,0,exists
                
    fileData.columns=['index','sigma_crown','sigma_max']                              #this is the csv's dataframe

    sigma_crown=fileData.sigma_crown.values #...[fileData.index==ind]
    sigma_max=fileData.sigma_max.values #...[fileData.index==ind] #get the maximum stress on the 3rd day

    stressData=pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)  #just need the stresses to be formatted in a way I can add to master DF
    stressData['sigma_crown']=sigma_crown
    stressData['sigma_max']=sigma_max
    stressData['Tf']=Tf
    stressData['dT']=dT
    
    return stressData,NsubSteps,exists

def getStressDF_P11(Tf,dT):

    fileString  =  'dataframes/p11_results/stresses_dframe_A617_'+str(Tf)+'Tf_'+str(dT)+'dT_0.4289R_30substeps_4days_14period_p11.csv'
    fileString2 =  'dataframes/p11_results/stresses_dframe_A617_'+str(Tf)+'Tf_'+str(dT)+'dT_0.4289R_20substeps_4days_14period_p11.csv'
    fileString3 =  'dataframes/p11_results/stresses_dframe_A617_'+str(Tf)+'Tf_'+str(dT)+'dT_0.4289R_40substeps_4days_14period_p11.csv'


    exists=True
    if os.path.exists(fileString):
        fileData=pd.read_csv(fileString)
        NsubSteps=30
    elif os.path.exists(fileString2):
        fileData=pd.read_csv(fileString2)
        NsubSteps=20
    elif os.path.exists(fileString3):
        fileData=pd.read_csv(fileString3)
        NsubSteps=40
    else:
        print('skipped', Tf, dT)
        exists=False
        return 0,0,exists
                
    fileData.columns=['index','sigma_crown','sigma_max']                              #this is the csv's dataframe

    sigma_crown=fileData.sigma_crown.values #...[fileData.index==ind]
    sigma_max=fileData.sigma_max.values #...[fileData.index==ind] #get the maximum stress on the 3rd day

    stressData=pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)  #just need the stresses to be formatted in a way I can add to master DF
    stressData['sigma_crown']=sigma_crown
    stressData['sigma_max']=sigma_max
    stressData['Tf']=Tf
    stressData['dT']=dT
    
    return stressData,NsubSteps,exists

def getStressDF_P12(Tf,dT):

    fileString  =  'dataframes/p12_results/stresses_dframe_800H_'+str(Tf)+'Tf_'+str(dT)+'dT_0.4617R_30substeps_4days_14period_p12.csv'
    fileString2 =  'dataframes/p12_results/stresses_dframe_800H_'+str(Tf)+'Tf_'+str(dT)+'dT_0.4617R_20substeps_4days_14period_p12.csv'
    fileString3 =  'dataframes/p12_results/stresses_dframe_800H_'+str(Tf)+'Tf_'+str(dT)+'dT_0.4617R_40substeps_4days_14period_p12.csv'


    exists=True
    if os.path.exists(fileString):
        fileData=pd.read_csv(fileString)
        NsubSteps=30
    elif os.path.exists(fileString2):
        fileData=pd.read_csv(fileString2)
        NsubSteps=20
    elif os.path.exists(fileString3):
        fileData=pd.read_csv(fileString3)
        NsubSteps=40
    else:
        print('skipped', Tf, dT)
        exists=False
        return 0,0,exists
                
    fileData.columns=['index','sigma_crown','sigma_max']                              #this is the csv's dataframe

    sigma_crown=fileData.sigma_crown.values #...[fileData.index==ind]
    sigma_max=fileData.sigma_max.values #...[fileData.index==ind] #get the maximum stress on the 3rd day

    stressData=pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)  #just need the stresses to be formatted in a way I can add to master DF
    stressData['sigma_crown']=sigma_crown
    stressData['sigma_max']=sigma_max
    stressData['Tf']=Tf
    stressData['dT']=dT
    
    return stressData,NsubSteps,exists


def getStressDF_dcStudy(Tf,dT,daysPt1,t_op2):

    fileString='dataframes/dcStudy/stresses_dframe_A230_'+str(Tf)+'Tf_'+str(dT)+'dT_0.5R_30substeps_'+str(daysPt1)+'daysPt1_1daysPt2_14period_vtopLast1_'+str(t_op2)+'hrs.csv'
    exists=True
    if os.path.exists(fileString):
        fileData=pd.read_csv(fileString)
        NsubSteps=30
    else:
        print('skipped', Tf, dT)
        exists=False
        return 0,0,exists
                
    fileData.columns=['index','sigma_crown','sigma_max']                              #this is the csv's dataframe

    sigma_crown=fileData.sigma_crown.values #...[fileData.index==ind]
    sigma_max=fileData.sigma_max.values #...[fileData.index==ind] #get the maximum stress on the 3rd day

    stressData=pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)  #just need the stresses to be formatted in a way I can add to master DF
    stressData['sigma_crown']=sigma_crown
    stressData['sigma_max']=sigma_max
    
    return stressData,NsubSteps,exists

def addInterpPt( xNew, Tf, masterDF,modeString):
    '''
    interpolates a creep damage or strain range based on existing database and adds to that db. For out of bounds cases, provides the first or last value
    xArray - array of deltaTs (C)
    yArray - array of creep or strainranges corresponding with the xArray positions (-)
    xNew   - desired interpolation point
    Tf     - fluid temperature of x/yArray (C)
    masterDf- the existing dataframe containing original datapoints
    modeString - tells function to make column name dc or strainRange3. datatype: string
    '''
    subSet=masterDF[masterDF.Tf==Tf].sort_values(by='dT')
    if modeString=='strainRange3':
        yArray=subSet.strainRange3.to_numpy()
    elif modeString=='dc':
        yArray=subSet.dc.to_numpy()
    else: 
        print('wrong modeString input')
    
    xArray=subSet.dT.to_numpy()
    f=spInt.interp1d(xArray, yArray,fill_value=(yArray[0], yArray[-1]),bounds_error=False)
    yNew=f(xNew)

    subDF=pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)  #just need the stresses to be formatted in a way I can add to master DF
    if modeString=='strainRange3':
        subDF['strainRange3']=np.array([ yNew ])
    elif modeString=='dc':
        subDF['dc']=np.array([ yNew ])
    else: 
        print('modeString not recognized')
    subDF['Tf']=np.array([ Tf ])
    subDF['dT']=np.array([ xNew ])
    
    newMasterDF=pd.concat([masterDF,subDF],ignore_index=True)

    return newMasterDF

def getPeakOpStress(stresses,period,nSubsteps,tOff):
    """
    returns the indices at which maximum operational stress occurs
    stresses - (MPa) df or array of vm mises stresses from SRLIFE simulation
    period   - (hrs) the operation period of each day
    tOff     - (hrs) the cutoff distance between minimas
    """

    
    min_inds = sPsig.find_peaks( stresses*(-1), distance=tOff*nSubsteps, width=tOff*nSubsteps/2 )[0]
    min_inds = np.concatenate((np.array([0]),min_inds))
    

    days=int(np.size(stresses)/nSubsteps/period)
    opMax_inds = np.array([])
    for day in range(days):
        left=day*period*nSubsteps
        right=left+period*nSubsteps+1
        min_set = min_inds[np.where( (min_inds[:] >= left) & (min_inds[:] <= right) )[0]]
        opMax_ind=np.argmax(stresses[min_set[0]:min_set[1]])+min_set[0]
        opMax_inds=np.concatenate( (opMax_inds,np.array([opMax_ind])) )
    fig,ax = plt.subplots()
    ax.plot(stresses,linewidth=1)
    ax.scatter(min_inds,stresses[min_inds],s=5,color='r',label='minimum points')
    ax.scatter(opMax_inds,stresses[opMax_inds],s=5, color='k',label='maximum points')
    # plt.show()
    plt.close()
    return opMax_inds
