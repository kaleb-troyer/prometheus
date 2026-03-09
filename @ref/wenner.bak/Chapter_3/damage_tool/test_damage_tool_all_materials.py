"""
made by jwenner to test speed, accuracy, and integrity of final version of damage_tool for all materials
"""

import damage_tool
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    ## make some 30 and 80 year contours, see what the vertical and horizontal difference is
    dmg_obj_80yr =damage_tool.damageTool('A230',interp_mode='LNDI') 
    dmg_obj_80yr.make_contour_function_from_interpolator(LTE_desired=80, cutoff=None, show_LTE_ctr=False)

    dmg_obj_30yr =damage_tool.damageTool('A230',interp_mode='LNDI') 
    dmg_obj_30yr.make_contour_function_from_interpolator(LTE_desired=30, cutoff=None, show_LTE_ctr=False)

    # get average dT diff
    Tf_range    =np.linspace(315,565)
    dTs_80yr    =dmg_obj_80yr.dT_function(Tf_range)
    dTs_30yr    =dmg_obj_30yr.dT_function(Tf_range)
    ##

    ## create a dmg tool instance for a specified material
    dmg_obj =damage_tool.damageTool('A230',interp_mode='LNDI') # LNDI is superior to the rbf method, I think
    dmg_obj.R=0.50
    
    ## select a few points in the thermal operating space
    # # test A617
    # dT_points   =np.array([105,215,172,180,210,160])
    # Tf_points   =np.array([540,350,475,540,540,445])
    # R_points    =dmg_obj.R*np.ones(dT_points.size)

    # # test A230
    dT_points   =np.array([105,215,172,170,210,160,276])
    Tf_points   =np.array([540,350,475,540,540,445,290])
    R_points    =dmg_obj.R*np.ones(dT_points.size)

    # # # test A282
    # dT_points   =np.array([105,215,185,170,210,160])
    # Tf_points   =np.array([540,350,495,540,540,445])
    # R_points    =dmg_obj.R*np.ones(dT_points.size)

    # # # test 740H
    # dT_points   =np.array([105,215,185,170,210,160])
    # Tf_points   =np.array([540,350,495,540,540,445])
    # R_points    =dmg_obj.R*np.ones(dT_points.size)
    
    # # test 316H
    # dT_points   =np.array([105,130,150,160,170,175])
    # Tf_points   =np.array([300,325,350,475,450,499])
    # R_points    =dmg_obj.R*np.ones(dT_points.size)

    # # # test 800H
    # dT_points   =np.array([105,130,105,151,175])
    # Tf_points   =np.array([490,460,450,460,330])
    # R_points    =dmg_obj.R*np.ones(dT_points.size)

    ## test the unrestrained LTE estimator - doesn't consider corrosion or SR
    LTEs_simple                 =dmg_obj.get_LTEs_no_limits(dT_points,Tf_points,R_points)
    dmg_obj.plot_dmg_map(op_dTs=dT_points,op_Tfs=Tf_points,LTEs=LTEs_simple)

    ## test the restrained LTE estimator - returns 0's for corrosion or SR points
    LTEs_careful                 =dmg_obj.get_LTEs(dT_points,Tf_points,R_points)
    dmg_obj.plot_dmg_map(op_dTs=dT_points,op_Tfs=Tf_points,LTEs=LTEs_careful)

    ## test the penalized LTE estimator - returns negative values based location
    LTEs_penalized                =dmg_obj.get_LTEs_w_penalties(dT_points,Tf_points,R_points)
    dmg_obj.plot_dmg_map(op_dTs=dT_points,op_Tfs=Tf_points,LTEs=LTEs_penalized)

    ## test a couple of design contours: 30, 80, 300
    # dmg_obj.make_contour_function_from_interpolator(LTE_desired=30, show_LTE_ctr=True)
    # dmg_obj.make_contour_function_from_interpolator(LTE_desired=80, show_LTE_ctr=True)
    # dmg_obj.make_contour_function_from_interpolator(LTE_desired=300, show_LTE_ctr=True)

    ## plot 30 year design contours for each material
    Tf_ctrs =np.linspace(290,565)

    dmg_obj_230 =damage_tool.damageTool('A230',interp_mode='LNDI') 
    dmg_obj_617 =damage_tool.damageTool('A617',interp_mode='LNDI') 
    dmg_obj_282 =damage_tool.damageTool('A282',interp_mode='LNDI') 
    dmg_obj_740 =damage_tool.damageTool('740H',interp_mode='LNDI') 

    dmg_obj_230.make_contour_function_from_interpolator(LTE_desired=100, show_LTE_ctr=True)
    dTs_230     =dmg_obj_230.dT_function(Tf_ctrs)
    dmg_obj_617.make_contour_function_from_interpolator(LTE_desired=100, show_LTE_ctr=True)
    dTs_617     =dmg_obj_617.dT_function(Tf_ctrs)
    dmg_obj_282.make_contour_function_from_interpolator(LTE_desired=100, show_LTE_ctr=True)
    dTs_282     =dmg_obj_282.dT_function(Tf_ctrs)
    dmg_obj_740.make_contour_function_from_interpolator(LTE_desired=100, show_LTE_ctr=True)
    dTs_740     =dmg_obj_740.dT_function(Tf_ctrs)

    dTs_corr_lowR,Tfs_corr_lowR   =dmg_obj_617.materialLimits.get_plot_pts(R=0.25)
    dTs_corr_highR,Tfs_corr_highR   =dmg_obj_617.materialLimits.get_plot_pts(R=0.75)

    fig,ax      =plt.subplots()
    fontsize    =14
    ax.plot(dTs_230,Tf_ctrs,label='A230',color='dodgerblue')
    ax.plot(dTs_617,Tf_ctrs,label='A617',color='black')
    ax.plot(dTs_282,Tf_ctrs,label='A282',color='olive')
    ax.plot(dTs_740,Tf_ctrs,label='740H',color='darkviolet')
    ax.plot(dTs_corr_lowR,Tfs_corr_lowR,label='Tcrit,R=0.25',linestyle='--',color='black')
    ax.plot(dTs_corr_highR,Tfs_corr_highR,label='Tcrit,R=0.75',linestyle='--',color='gray')
    ax.set_xlim(50,300)
    ax.set_ylim(290,565)
    ax.legend(fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-2)
    ax.set_xlabel('total temperature difference (C)',fontsize=fontsize)
    ax.set_ylabel('fluid temperature (C)',fontsize=fontsize)
    fig.savefig('imgs/the_big_four_contours_100yrs',dpi=300)
    plt.show()



