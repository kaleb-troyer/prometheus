"""
created by jwenner on 12/4/2025 to create lifetime profiles for Crescent Dunes and output other interesting information
"""
import damage_tool
import sys
sys.path.append('../thermal_and_optical_tools/')
import helpers_thermal_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import numpy as np

## adapted from helpers_thermal_model
def plot_results(W, H, Npanels, ntubesim, nz, results, label_name, savename='default_receiver_heatmap', vmin=0, vmax=1):
    """
    makes a heatmap using imshow, based on receiver width and height. Resolution is dependent on number of tubes/panel
    --
    results - a nz x npanels x ntubes/panel size array of whatever result you want to plot
    """
    cbr_fontsize =12
    fig,ax=plt.subplots(tight_layout=True)
    if label_name == 'lifetimes (yrs)' or label_name == 'log10(lifetimes (yrs))':
        cmap_choice ='plasma_r'
    else: 
        cmap_choice ='inferno'
    im = ax.imshow(results.reshape(nz,Npanels*ntubesim), extent=[-W/2, W/2, 0, H], vmin=vmin, vmax=vmax, cmap=cmap_choice )
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('y position (m)')
    divider = make_axes_locatable(ax)  
    cax = divider.append_axes("right", size="2%", pad=0.08)     # thanks stack exchange!
    cbr = fig.colorbar(im, cax=cax)
    cbr.set_label(label=label_name, size=cbr_fontsize)
    cbr.ax.tick_params(labelsize=cbr_fontsize)
    fig.savefig(f'imgs/{savename}',dpi=300)
    plt.show()
    plt.close(fig)
    return 
##

if __name__ == '__main__':
    ## receiver imputs from crescent_dunes.json
    H       =18.59
    D       =15.2*math.pi   # need the total unrolled circumference of the receiver
    Npanels =14
    ntubesim=3
    nz      =50

    ## specify the filename
    mat         ='A230'
    if mat in ['A230','A282','A617']:
        f_mat   ='A230'     # only thermal model property options in thermal model were A230 or 740H
    # elif mat =='A617':    # tried this but found a very similar result to using 740H thermal results for SS316. bc janna's conductivity model for SS316 is lower than EES predicts
    #     f_mat   ='SS316'
    else:
        f_mat   =mat
    filename    =f'reports/damage_tool_CD_study_{f_mat}.json'

    ## make a damage object
    dmg_obj     =damage_tool.damageTool(mat)

    ## read in results
    qabs    =helpers_thermal_model.json_to_np(filename,'qabs')
    qsinc   =qabs/0.96                                              # original model used  absorptivity of 0.96
    dTs     =helpers_thermal_model.json_to_np(filename,'dTs')
    Tfs     =helpers_thermal_model.json_to_np(filename,'Tfs')
    Rs      =helpers_thermal_model.json_to_np(filename,'Rs')
    LTEs    =dmg_obj.get_LTEs(dTs.flatten(),Tfs.flatten(),Rs.flatten())
    min_panel_LTEs, tube_min_LTEs =dmg_obj.calc_minimum_panel_LTEs_simple_inputs(ntubesim, nz, Npanels, LTEs)
    print(f'minimum LTE is {LTEs.min():.2f} at {Tfs.flatten()[np.argmin(LTEs)]:.2f} Tf and {dTs.flatten()[np.argmin(LTEs)]:.2f} dT')
    print(f'predicted number of sub 30 year panels is:{np.sum(min_panel_LTEs < 30)}')
    # ### plot the incident flux profile
    # plot_results(D, H, Npanels, ntubesim, nz, qsinc, 'incident flux (kW/m2)', savename=f'flux_heatmap_{mat}', vmin=1000, vmax=0) # if you specifically want lifetimes, use 'lifetimes (yrs)'
    # ### plot the fluid temperatures
    # plot_results(D, H, Npanels, ntubesim, nz, Tfs, 'fluid temperature (C)', savename=f'fluid_temp_heatmap_{mat}', vmin=565, vmax=290) # if you specifically want lifetimes, use 'lifetimes (yrs)'
    # ### plot the delta Ts
    # ## 740H: 0,220, A230: similar
    # plot_results(D, H, Npanels, ntubesim, nz, dTs, 'total temperature difference (C)', savename=f'dT_heatmap_{mat}', vmin=0, vmax=220) # if you specifically want lifetimes, use 'lifetimes (yrs)'
    
    ### plot the LTEs
    # 740H
    # vmins: 740H:none
    # plot_results(D, H, Npanels, ntubesim, nz, np.log10(LTEs), 'log10(lifetimes (yrs))', savename=f'LTE_heatmap_{mat}', vmin=None, vmax=None) # use for 740H. if you specifically want lifetimes, use 'lifetimes (yrs)' or 'log10(lifetimes (yrs))'
    # plot_results(D, H, Npanels, ntubesim, nz, LTEs, 'lifetimes (yrs)', savename=f'LTE_heatmap_{mat}', vmin=None, vmax=100) # use for A230. if you specifically want lifetimes, use 'lifetimes (yrs)' or 'log10(lifetimes (yrs))'
    # plot_results(D, H, Npanels, ntubesim, nz, LTEs, 'lifetimes (yrs)', savename=f'LTE_heatmap_{mat}', vmin=None, vmax=100) # use for A617. if you specifically want lifetimes, use 'lifetimes (yrs)' or 'log10(lifetimes (yrs))'
    # plot_results(D, H, Npanels, ntubesim, nz, np.log10(LTEs), 'log10(lifetimes (yrs))', savename=f'LTE_heatmap_{mat}', vmin=None, vmax=None) # use for A282. if you specifically want lifetimes, use 'lifetimes (yrs)' or 'log10(lifetimes (yrs))'

    dmg_obj.plot_dmg_map(include_ratios=False, op_dTs=dTs.flatten(), op_Tfs=Tfs.flatten(), savename=f'CD_thermal_points_dmg_map_{mat}' )

