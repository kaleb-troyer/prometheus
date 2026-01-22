"""
created 7/15/25 by jwenner
reads pre-generated csv and plots on a damage map 
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
#import material_limits

def fmt(x):
    s = f"{x:.0f}" + ' (yrs)'
    return s


def plot(DMG_table,mat,days_in_year,case_name):
    ## plotting details
    fig,ax=plt.subplots(tight_layout=True)
    fontsize=10

    ### get the maximum damage ratio for all parametric cases

    # ### make pivot table for ratio UNDEVELOPED/COMMENTED OUT BECAUSE NO SIGNIFICANT RATIOS
    # DMG_table["dmg_ratio_log"]=np.log10(dmgDF_parent["drat"])
    # df_HeatMap=dmgDF_parent.pivot(index="Tinner",columns='deltaT',values='drat_log') #re-organize data for countour plots
    # df_HeatMap = df_HeatMap.sort_index(ascending=False) #flip it
    # drats=pd.DataFrame.to_numpy(df_HeatMap)
    # # levels = [-6,-5,-4,-3,-2,-1,0,1,2] #can be used in ctf line
    # levels = [-6,-5,-4,-3,-2,-1] # this is for A617 and 740H and 800H
    # # levels = [-3,-2,-1,0,1,2]
    # DesRegionY = df_HeatMap.index.to_numpy()
    # DesRegionX = df_HeatMap.columns.to_numpy()
    # ctf=ax.contourf(DesRegionX, DesRegionY, drats,cmap=cm.YlGn,levels=levels) # removed levels=levels
    # cb1=fig.colorbar(ctf)
    # cb1.set_label(label='log($d_f$/$d_c$)',fontsize=fontsize)
    # cb1.ax.tick_params(labelsize=fontsize)

    # ### add a set of material limits
    # R=0.5 # sets the assumed resistance ratio 
    # MAT_LIM = material_limits.MaterialLimits(mat) # creating a class of material limits for specific material
    # x_LIM_points, y_LIM_points = MAT_LIM.get_plot_pts(R=R)
    # # ax.plot(x_LIM_points,y_LIM_points,color='slateblue')
    # y_line_pts=np.ones(x_LIM_points.size)*600 # just need an upper bound to fill between
    # ax.fill_between(x_LIM_points,y_LIM_points,y_line_pts,color='rosybrown',zorder=2)
    # ax.text(167,540,'C.R.',fontsize=fontsize-3) # 167 x coord for A230 # 175 for some other alloy

    ## plot SR zone
    try:
        fileName = 'SRregion_'+str(mat)+'.csv' #save before flipping
        SRzone=pd.read_csv( ('dataframes/'+fileName) , index_col=0)
        SRzone = SRzone.sort_index(ascending=False) #flip it
        SRzoneVals=pd.DataFrame.to_numpy(SRzone)
        SRlevels = (2.00,4.00)
        DesRegionY = SRzone.index.to_numpy()
        DesRegionX = SRzone.columns.to_numpy().astype(float) #np.linspace(40,370,SR_RES[mat])#SRzone.columns.to_numpy() #<- i don't think we need this anymore
        ctfSR=ax.contourf(DesRegionX, DesRegionY, SRzoneVals,levels=SRlevels,cmap=cm.inferno,zorder=2) # removed levels=levels
        ax.text(200,540,'S.R.R.',fontsize=fontsize-3)
    except FileNotFoundError:
        print('no SR information found')

    ### make another pivot table for total damage
    if (mat == '316H') or (mat == '800H'): 
        ax.set_xlim(50,300)
        ax.set_ylim(bottom=275,top=575)
    elif mat=='A230':
        ax.set_xlim(50,300)
        ax.set_ylim(bottom=275,top=575)
    else:
        ax.set_xlim(50,300)
        ax.set_ylim(bottom=275,top=575)

    df_ltes=DMG_table.pivot(index="Tf",columns="dT",values="lte_cycles")
    df_ltes=df_ltes.sort_index(ascending=False) #flip it
    LTEs_years=pd.DataFrame.to_numpy(df_ltes)/days_in_year # make a countourf plot using matplotlib
    levels=(5,10,15,20,30,40,50,60,70,80)#range(0,80,10) #can be used in ctf line
    y2 = df_ltes.index.to_numpy()
    x2 = df_ltes.columns.to_numpy()
    ctf=ax.contour(x2, y2, LTEs_years,levels=levels,cmap=cm.copper_r,zorder=1) #was using cm.cool or colors='black', levels=levels
    #cb2=plt.colorbar()


    ### manually place contour
    ax.axes.clabel(ctf,inline=True,inline_spacing=1, fontsize=6, fmt=fmt,manual=True,levels=levels) #left click: place. center click: done


    ### final touches on plot
    ax.grid(visible=False)
    ax.set_facecolor('white')
    ax.spines['left'].set_color('black')  #use this line if plotting only one at a time
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.tick_params(bottom=True, top=False, left=True, right=False)
    ax.set_xlabel('$\Delta T$ [C]',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    ax.set_ylabel('$T_{f}$ [C]',fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xlabel('Difference in Crown Temperature Relative to Fluid Temperature (°C)')
    ax.set_ylabel('Fluid Temperature (°C)')
    plt.savefig(dpi=300,fname=(case_name+'/'+case_name+'_'+mat+'_dmg_map.png'))
    plt.show()
