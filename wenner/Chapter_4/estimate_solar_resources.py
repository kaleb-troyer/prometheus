"""
created by jwenner on 11/6/2025 to estimate the average DNI for every hour of representative days in the 4 seasons
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator 
import numpy as np
import json

if __name__ == "__main__":
    df =pd.read_csv('weather_file.csv')
    df =pd.read_csv('weather_file.csv')
    jan_DNI         =df[df.Month==1]
    jan_DNI_prof    =jan_DNI.pivot_table(index='Hour', values='DNI', aggfunc='mean')
    
    feb_DNI         =df[df.Month==2]
    feb_DNI_prof    =feb_DNI.pivot_table(index='Hour', values='DNI', aggfunc='mean')


    mar_DNI         =df[df.Month==3]
    mar_DNI_prof    =mar_DNI.pivot_table(index='Hour', values='DNI', aggfunc='mean')
    
    apr_DNI         =df[df.Month==4]
    apr_DNI_prof    =apr_DNI.pivot_table(index='Hour', values='DNI', aggfunc='mean')
    
    may_DNI         =df[df.Month==5]
    may_DNI_prof    =may_DNI.pivot_table(index='Hour', values='DNI', aggfunc='mean')

    jun_DNI         =df[df.Month==6]
    jun_DNI_prof    =jun_DNI.pivot_table(index='Hour', values='DNI', aggfunc='mean')

    jul_DNI         =df[df.Month==7]
    jul_DNI_prof    =jul_DNI.pivot_table(index='Hour', values='DNI', aggfunc='mean')

    aug_DNI         =df[df.Month==8]
    aug_DNI_prof    =aug_DNI.pivot_table(index='Hour', values='DNI', aggfunc='mean')

    sep_DNI         =df[df.Month==9]
    sep_DNI_prof    =sep_DNI.pivot_table(index='Hour', values='DNI', aggfunc='mean')

    oct_DNI         =df[df.Month==10]
    oct_DNI_prof    =oct_DNI.pivot_table(index='Hour', values='DNI', aggfunc='mean')

    nov_DNI         =df[df.Month==11]
    nov_DNI_prof    =nov_DNI.pivot_table(index='Hour', values='DNI', aggfunc='mean')

    dec_DNI         =df[df.Month==12]
    dec_DNI_prof    =dec_DNI.pivot_table(index='Hour', values='DNI', aggfunc='mean')

    # calculate seasonal DNI
    spring_DNI_prof =(feb_DNI_prof.values+mar_DNI_prof.values+apr_DNI_prof.values)/3
    summer_DNI_prof =(may_DNI_prof.values+jun_DNI_prof.values+jul_DNI_prof.values)/3
    fall_DNI_prof   =(aug_DNI_prof.values+sep_DNI_prof.values+oct_DNI_prof.values)/3
    winter_DNI_prof =(nov_DNI_prof.values+dec_DNI_prof.values+jan_DNI_prof.values)/3

    report_dict={'spring':spring_DNI_prof.tolist(), 'summer':summer_DNI_prof.tolist(), 'fall':fall_DNI_prof.tolist(), 'winter':winter_DNI_prof.tolist()}
    ## save dataframes
    with open('seasonal_DNI', "w") as f:
        json.dump(report_dict, f)

    ## plot the average DNI for dec, mar, june, and sep
    fontsize    =14
    fig,ax      =plt.subplots()
    # ax.plot(dec_DNI_prof.index+1, dec_DNI_prof.values, label='Dec.', marker='.')
    # ax.plot(mar_DNI_prof.index+1, mar_DNI_prof.values, label='Mar.', marker='.')
    # ax.plot(jun_DNI_prof.index+1, jun_DNI_prof.values, label='June', marker='.')
    # ax.plot(sep_DNI_prof.index+1, sep_DNI_prof.values, label='sept.', marker='.')
    ax.plot( np.arange(spring_DNI_prof.size)+1, spring_DNI_prof, label='spring', marker='.')
    ax.plot( np.arange(summer_DNI_prof.size)+1, summer_DNI_prof, label='summer', marker='.')
    ax.plot( np.arange(fall_DNI_prof.size)+1, fall_DNI_prof, label='fall', marker='.')
    ax.plot( np.arange(winter_DNI_prof.size)+1, winter_DNI_prof, label='winter', marker='.')
    ax.legend()
    ax.set_xlabel('hour of day', fontsize=fontsize)
    ax.set_ylabel('DNI (W/m2)',  fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    fig.savefig('imgs/average_seasonal_DNI', dpi=300)
    plt.show()
    plt.close()
    ## plot the representative optical simulation days
    fig, ax     =plt.subplots()
    ax.plot(mar_DNI.Hour.unique()+1, mar_DNI.DNI[mar_DNI.Day==20], label='spring equinox', marker='.')
    ax.plot(sep_DNI.Hour.unique()+1, sep_DNI.DNI[sep_DNI.Day==23], label='fall equinox', marker='.')
    ax.plot(dec_DNI.Hour.unique()+1, dec_DNI.DNI[dec_DNI.Day==21], label='winter solstice', marker='.')
    ax.plot(jun_DNI.Hour.unique()+1, jun_DNI.DNI[jun_DNI.Day==22], label='summer solstice', marker='.')
    ax.set_xlabel('hour of day', fontsize=fontsize)
    ax.set_ylabel('DNI (W/m2)', fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.legend()
    fig.savefig('imgs/DNI_profiles', dpi=300)
    plt.show()
    plt.close()