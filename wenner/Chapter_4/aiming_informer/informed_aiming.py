"""
created by jwenner on 8/1/2025
module that implements available damage tools, thermal models, etc to generate a set of aimpoints and advise designs
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib
import damage_tool
import tube_jwenner
import json
import scipy.optimize as scOpt

def flux_from_Tf(dT_fun, Tfs, tube):
    """
    helper function designed to output the allowable flux (W/m2) based on the allowable wall temperature difference
    ---
    dT_fun      - (C) contour function
    Tfs         - (C) array of fluid temepratures at the desired dT locations. 
    tube        - (thermal model object) used to get internal HTC and other useful properties
    ---
    returns: an absorbed flux array for all input points
    """
    dTs         =dT_fun(Tfs)
    # get the wall conductivities
    R_assume        =0.5
    T_crown_outs    =Tfs+dTs
    T_crown_ins     =Tfs+dTs/(1+R_assume)
    T_wall_effs     =T_crown_ins
    ks              =tube.tube_wall.k(T_wall_effs+273.15)

    # get the heat transfer coefficient assuming the tube mass flow is current
    Tfs_K               =Tfs + 273.15
    T_crown_in_Ks       =T_crown_ins+ 273.15 
    Tsafety             =50
    HTCs, REs, Vels     =tube.internal_h(Tfs_K, wall_visc=True, Tw=T_crown_in_Ks-Tsafety)   # input to internal_h is in Kelvin in NREL model

    # finally get around to calculating flux!
    fluxes_abs_crown_W  =dTs/( (tube.twall/ks) + (1/HTCs) )

    return fluxes_abs_crown_W

def offset_region_ebalance(Tf_out, m_dot, W_panel, L_act, dT_fun, tube, Tf_in):
    """
    built for use in fsolve. Returns the difference betwen the LHS and RHS of upstream offset region energy balance for any given Tf
    """
    Qdot_section    =W_panel*(1/2)*L_act*flux_from_Tf(dT_fun, Tf_out, tube)
    Tf_avg          =(Tf_in+Tf_out)/2
    cp              =tube.fluid.cp(Tf_avg + 273.15)
    res = Tf_out - Qdot_section/(m_dot*cp) - Tf_in
    return res

def generate_ideal_fluxmap(dmg_case, tube, LTE_desired, mflow, W_panel, L_t, flowpath_config, show=False, cutoff=270):
    """
    output what the lowest required area is for a given desired lifetime, material, and aspect ratio. Also plots 
    the resultant flux profile
    --
    dmg_case        - (damage tool object) damage tool instance
    tube            - (thermal model object) used to get internal heat transfer coefficients
    LTE_desired     - (int) desired lifetime of the receiver in years
    cp              - (J/kg-K) specific heat capacity of the fluid
    k               - (W/m-K) metal conductivity
    mflow           - (kg/s) total mass flow
    W_panel         - (float) the panel width
    L_tube          - the set tube height, i.e. the receiver's height
    flowpath_config - (str). Receiver's flowpath. Valid strings currently include: '2_ctr'
    img_name        - name to save the flux profile to
    show            - boolean. Makes decision to show contour or not
    ---
    returns:
    ideal_flux_profile - (array, kW/m2) Npanel x  array of spatial flux values
    A_req              - (float, m2) maximum required area, including extraneous panel-finishing sections
    ideal_flux_img     - (imshow - generated image of kW/m2 flux distribution)
    """
    ##  load a contour by calling the damage case (damage tool instance) and specifying the desired LTE
    dmg_case.make_contour_function_from_interpolator(LTE_desired, cutoff=cutoff, show_LTE_ctr=show)
    # dmg_case.make_contour_function(LTE_desired)   # old method was creating a contour that was slightly off the RBF contour

    ##  calculate the mass flow for each flowpath
    if flowpath_config == '2_ctr':
        npaths      =2
        is_symmetric=True
        x_start     =0
        y_start     =L_t
    mflow_per_path = mflow/npaths   # (kg/s)
    ##  calculate the maximum allowable flux for every fluid temperature degree
    # get the dT at every degree
    Trise_inc =0.1
    Tf_points =np.arange(290,565,Trise_inc)
    print('assuming typical 290-565 C fluid temperature range')
    dT_points =dmg_case.dT_function(Tf_points)
    # get the wall conductivities
    R_assume    =0.5
    T_crown_out =Tf_points+dT_points
    T_crown_in  =Tf_points+dT_points/(1+R_assume)
    T_wall_eff  =T_crown_in #(T_crown_in+T_crown_out)/2
    ks          =tube.tube_wall.k(T_wall_eff+273.15)
    # get the heat transfer coefficient
    ntubes_panel    =int(W_panel/tube.OD)
    mflow_tube      =mflow/npaths/ntubes_panel
    tube.mflow      =mflow_tube
    Tf_points_K     =Tf_points + 273.15
    T_crown_in_K    =T_crown_in+ 273.15
    print('estimating heat transfer coefficient')
   
    Tsafety         =50
    HTCs, REs, Vels = tube.internal_h(Tf_points_K, wall_visc=True, Tw=T_crown_in_K-Tsafety)   # input to internal_h is in Kelvin in NREL model

    # ## plot that explains the temporary increase in allowable flux seen early on in the flowpath
    # fig,ax =plt.subplots(tight_layout=True)
    # ax.plot(Tf_points,dT_points)
    # ax2=ax.twinx()
    # R_per_As =( (tube.twall/ks) + (1/HTCs) )
    # ax2.plot(Tf_points,R_per_As,color='k')
    # ax.text(365,150, 'R$_{crown}$/A')
    # ax.text(400,225, '$\Delta T$')
    # ax.set_xlabel('fluid temperature (C)')
    # ax.set_ylabel('temperature difference (C)')
    # ax2.set_ylabel('crown resistance (K/W-m$^2$)')
    # plt.savefig('imgs/why_allowable_flux_increases',dpi=300)
    # plt.show()
    # plt.close()
    # ##

    # finally get around to calculating flux!
    fluxes_abs_crown_W =dT_points/( (tube.twall/ks) + (1/HTCs) )
    fluxes_inc_crown_W  =fluxes_abs_crown_W/tube.solar_abs
    ##  calculate the power required to raise the fluid temperature by 1 degree
    CPs =tube.fluid.cp(Tf_points + 273.15)
    Qdots_incr =mflow_per_path*CPs*Trise_inc   # (W)
    ##  calculate the required area for every fluid temperature degree
    Areas   =Qdots_incr/fluxes_abs_crown_W
    delta_Ls=Areas/W_panel
    ##  make the receiver flux dictionary
    rec_flux_dict =assign_fluxes_to_panels(x_start, y_start, delta_Ls, W_panel, L_t, fluxes_inc_crown_W)

    ## calculate the receiver power
    if flowpath_config == '2_ctr':
        Q_inc_rec  =np.sum(Areas*fluxes_inc_crown_W)*2 # calculate total incident power on the receiver
        A_req      =np.sum(Areas)*2
    else:
        print('flowpath not recognized - could not calculate incident power')

    return rec_flux_dict, Q_inc_rec, Areas, A_req

def generate_ideal_fluxmap_with_offset(dmg_case, tube, LTE_desired, mflow, W_panel, L_t, v_offset, flowpath_config, show_LTE_ctr=False, cutoff=270):
    """
    output what the lowest required area is for a given desired lifetime, material, panel width, and required edge offset.
    --
    dmg_case        - (damage tool object) damage tool instance
    tube            - (thermal model object) used to get internal heat transfer coefficients
    LTE_desired     - (int) desired lifetime of the receiver in years
    cp              - (J/kg-K) specific heat capacity of the fluid
    k               - (W/m-K) metal conductivity
    mflow           - (kg/s) total mass flow
    W_panel         - (float) the panel width
    L_t             - the set tube height, i.e. the receiver's height
    v_offset        - (m) distance between the last allowed aiming point and the edge of the receiver
    flowpath_config - (str). Receiver's flowpath. Valid strings currently include: '2_ctr'
    ---
    returns:
    ideal_flux_profile - (array, kW/m2) Npanel x  array of spatial flux values
    A_req              - (float, m2) maximum required area, including extraneous panel-finishing sections
    ideal_flux_img     - (imshow - generated image of kW/m2 flux distribution)
    """
    ## generate the lifetime contour
    ##  load a contour by calling the damage case (damage tool instance) and specifying the desired LTE
    dmg_case.make_contour_function_from_interpolator(LTE_desired, cutoff=cutoff, show_LTE_ctr=show_LTE_ctr)

    ## get the heat transfer problem information
    #  calculate the mass flow for each flowpath
    if flowpath_config == '2_ctr':
        npaths      =2
        is_symmetric=True
        x_start     =0
        y_start     =L_t
    mflow_per_path  =mflow/npaths       # (kg/s)
    Tf_start        =290                # (C) starting fluid temperature 
    Tf_end          =565                # (C) target outlet fluid temperature
    print(f'assuming typical {Tf_start}-{Tf_end} C fluid temperature range')

    # update the tube mass flow
    ntubes_panel    =int(W_panel/tube.OD)
    mflow_tube      =mflow/npaths/ntubes_panel
    tube.mflow      =mflow_tube
  
    #### sequentially add sections until the flowpath temperature is reached. Sections are defined by a flux function ---- flux =m*(ypt-a)+b
    rec_flux_dict   ={}     # receiver dictionary containing each panel dictionary
    xpt             =x_start    
    ypt             =y_start    
    xpts            =[]     # all the left x locations of panels
    ypts            =[]     # all the upstream y locations for flux sections
    delta_const     =-1     # constant responsible for forwarding or reversing the flow. (-1) is down
    Tf_pt           =Tf_start # set the state tracker fluid temperature to the starting temperature
    Q_inc_rec       =0
    Areas           =[]

    while Tf_pt < Tf_end: # every iteration adds a panel to the receiver dictionary
        # initialize values for panel dictionary
        panel_flux_funcs        =[]     # (W/m2) a list of each assigned section's incident flux lambda function. Open function on downstream side
        ypts                    =[]
        panel_dict              ={}
        is_at_panel_end         =False    
        
        while is_at_panel_end == False: # every iteration adds a section to the panel's flowpath
            
            # check if this is the last flux and not at end of panel. if so, need to adjust!
            if Tf_pt > Tf_end and not is_at_panel_end:
                y_panel_end =0 if delta_const == -1 else L_t
                flux_func   =lambda y, a=0, ypt=ypt, y_panel_end=y_panel_end: 0 + a  if (ypt <= y < y_panel_end) or (y_panel_end < y <= ypt) else None# just adding a with default value to ensure that lambda function is uniquely stored in memory
                panel_flux_funcs.append(flux_func)
                ypts.append(ypt)
                break
            
            ### determine what case this section is - starting offset region, ending offset region, normal region, or error
            ## case 1 & 2: inside an offset region, so we need to decide which one it is
            if (L_t-v_offset < ypt <= L_t) or (0 <= ypt < 0+v_offset): 
                # downwards flow
                if (L_t-v_offset < ypt <= L_t) and delta_const == -1: # starting offset region (case 1)
                    is_at_ending_offset_region      =False
                    is_at_starting_offset_region    =True
                elif (0 <= ypt < 0+v_offset) and delta_const == -1: # ending offset region (case 2)
                    is_at_ending_offset_region      =True
                    is_at_starting_offset_region    =False
                # upwards flow cases
                elif (0 <= ypt < 0+v_offset) and delta_const == 1: # starting offset region case (case 1)
                    is_at_ending_offset_region      =False
                    is_at_starting_offset_region    =True
                elif (L_t-v_offset < ypt <= L_t) and delta_const == 1: # ending offset region case (case 2)
                    is_at_ending_offset_region      =True
                    is_at_starting_offset_region    =False

            ## case 3: outside the offset region, later we just solve for the next 1 C allowable flux, then backsolve for the required area
            elif v_offset <= ypt <= L_t-v_offset:
                is_at_ending_offset_region      =False
                is_at_starting_offset_region    =False

            ## case 4, something went wrong!
            else:
                print('y location out of bounds')
                quit()
            ###

            ### advance the Tf_pt variable towards outlet temperature by adding a section. Method depends on the section case
            ## case 1
            if is_at_starting_offset_region and not is_at_ending_offset_region:
                # iteratively solve for the allowable flux in the starting offset region so that the allowable flux is never exceeded (assuming limit function is linear in small segment)
                Tf_next=scOpt.fsolve(offset_region_ebalance,(Tf_pt+1),(mflow_per_path, W_panel, v_offset, dmg_case.dT_function, tube, Tf_pt))
                # get the flux from the solved Tf
                flux_abs_crown_W  =flux_from_Tf(dmg_case.dT_function, np.array(Tf_next), tube)[0]
                # build a lambda function to calculate the flux anywhere in the offset region
                if delta_const == -1: # downward flow case
                    m   =-(flux_abs_crown_W/tube.solar_abs)/v_offset
                    a   =L_t
                    b   =0
                else: # upward flow case
                    m   =(flux_abs_crown_W/tube.solar_abs)/v_offset
                    a   =0
                    b   =0
                # calculate the added heating from the offset region
                L_offset_actual =v_offset # when going upstream, we have exact control of the ending offset coordinate
                Q_section   =W_panel*(1/2)*L_offset_actual*(flux_abs_crown_W*tube.solar_abs) 
                # set the next ypt based on case
                ypt_next    =v_offset if delta_const == 1 else L_t-v_offset # delta_const ==1 is upwards flow
                # calculate the offset section area
                area        =W_panel*L_offset_actual
                # make the lambda function
                flux_func   = lambda y, m=m, a=a, b=b, ypt=ypt, ypt_next=ypt_next: m*(y-a) + b if (ypt <= y < ypt_next) or (ypt_next < y <= ypt) else None
            ## case 2
            elif not is_at_starting_offset_region and is_at_ending_offset_region:
                # use the previous flux to set the Qmax in this offset region
                if delta_const == -1: # downward flow case
                    m =panel_flux_funcs[-1](ypts[-1])/ypt
                    a =0
                    b =0
                else: # upward flow case
                    m =-panel_flux_funcs[-1](ypts[-1])/(L_t-ypt)
                    a =L_t
                    b =0
                # calculate the added heating from the offset region
                L_offset_actual =ypt if delta_const == -1 else L_t - ypt # the offset length might be slightly smaller than the v_offset region because of 1 C discretization
                Q_section   =W_panel*(1/2)*L_offset_actual*(panel_flux_funcs[-1](ypts[-1])*tube.solar_abs)    # integrating a linear flux profile over the panel width and actual offset length just yields W_panel*(triangle area)
                # calculate the offset section area
                area        =W_panel*L_offset_actual
                # set the next ypt based on case
                ypt_next    =L_t if delta_const == 1 else 0 # delta_const ==1 is upwards flow
                # make the lambda function. Note that the ending region piecewise function needs to include starting and ending point
                flux_func   = lambda y, m=m, a=a, b=b, ypt=ypt, ypt_next=ypt_next: m*(y-a) + b if (ypt <= y <= ypt_next) or (ypt_next <= y <= ypt) else None
                # we have now ended the panel
                is_at_panel_end =True

            ## case 3
            elif not is_at_starting_offset_region and not is_at_ending_offset_region:
                # set the flux based on the Tf_pt + dTf [C] allowable dT
                dTf             =0.1  # amount of desired fluid temperature rise per section
                Tf_next         =Tf_pt + dTf  # the next downstream fluid temperature
                flux_abs_crown_W  =flux_from_Tf(dmg_case.dT_function, np.array(Tf_next), tube)
                # calculate energy required in order to determine section length
                cp          =tube.fluid.cp(Tf_next + 273.15)
                Qdot_reqd   =mflow_per_path*cp*dTf   # (W)
                #  calculate the required area to achieve desired fluid temperature rise
                area        =Qdot_reqd/flux_abs_crown_W
                # calculate the required section length
                L_section   =area/W_panel
                # calculate the added heating from section (should match Qdot_reqd)
                Q_section   =area*flux_abs_crown_W
                # convert the absorbed flux into incident flux
                flux_inc_crown_W =flux_abs_crown_W/tube.solar_abs
                # set the next ypt based on the case
                ypt_next    =ypt+L_section if delta_const == 1 else ypt-L_section
                # make flatline lambda function
                flux_func   =lambda y, q_flux_max=flux_inc_crown_W, ypt=ypt, ypt_next=ypt_next: q_flux_max + 0*y if (ypt <= y < ypt_next) or (ypt_next < y <= ypt)  else None # stated as a lambda function because we need them for the offset regions. If we wanted, could be linear function in future

            ## update states, add values to lists
            panel_flux_funcs.append(flux_func)
            ypts.append(ypt)
            ypt =ypt_next
            # add the current section heating to the fluid
            cp_pt =tube.fluid.cp(Tf_pt + 273.15)
            Tf_pt   =( Q_section/(mflow_per_path*cp_pt) ) + Tf_pt 
            # track the total power
            Q_inc_rec   =Q_inc_rec + (Q_section/tube.solar_abs)
            # update the areas
            Areas.append(area)

        delta_const =delta_const*(-1) # change the flow direction

        # populate the panel dictionary
        panel_dict['fluxes']    =panel_flux_funcs
        panel_dict['ypts']      =ypts

        # add panel to receiver dictionary
        xpt_name                =f'{xpt:.2f}'
        rec_flux_dict[xpt_name] =panel_dict

        # move to the next panel
        xpt =xpt+W_panel
        xpts.append(xpt)
    
    ## calculate the total receiver power based on flowpath configuration
    if flowpath_config == '2_ctr':
        Q_inc_rec  =Q_inc_rec*2 # calculate total incident power on the receiver
        A_req      =np.sum(Areas)*2
    else:
        print('flowpath not recognized - could not calculate incident power')


    return rec_flux_dict, Q_inc_rec, Areas, A_req

def assign_fluxes_to_panels(x_start, y_start, y_deltas, W_panel, L_t, fluxes):
    """
    Makes a dictionary consisting of n panels, each with a varying sized list of fluxes. Panel keys are the left-most x-coordinate
        The ypt refer to the upstream y-coordinate of that section.
    ---
    x_start         - (m) x location for begining of flowpath aka left side of a panel in a right-going flowpath
    y_start         - (m) the flowpath's starting location in the first panel
    y_deltas        - (m, list) the flux section length
    W_panel         - (m) width of the panel. All panels assumed to have the same width
    L_t             - (m) tube length. Determines when to serpentine the flow.
    fluxes          - (kW/m2, list) the flux in each section. Size should match y_deltas
    """
    rec_flux_dict   ={}     # receiver dictionary containing each panel dictionary
    i               =0      # counter for accessing the correct flux
    xpt             =x_start    
    ypt             =y_start    
    xpts            =[]     # all the left x locations of panels
    ypts            =[]     # all the upstream y locations for flux sections
    delta_covered   =0      # accounts for spillage of a section onto a new panel
    delta_const     =-1     # constant responsible for forwarding or reversing the flow

    while i < len(fluxes)-1: # assign every flux a location
        panel_fluxes    =[]
        ypts            =[]
        panel_dict      ={}
        is_at_panel_end =False

        while is_at_panel_end == False:

            # check if this is the last flux and not at end of panel. if so, need to adjust!
            if (i == len(fluxes)) and not is_at_panel_end:
                panel_fluxes.append(0)
                ypts.append(ypt)
                break

            # add the fluxes to the panel's master list
            panel_fluxes.append(fluxes[i])
            # add the ypt to the panel's master list
            ypts.append(ypt)

            # calculate the next ypt, if delta_covered is nonzero then use once and reset immediately after
            ypt = ypt + (y_deltas[i]-delta_covered)*delta_const
            delta_covered =0    # the amount of the section length that exists on the previous panel 

            if ypt >= L_t: # the ypt exceeds the height of the receiver, need to switch directions
                is_at_panel_end =True
                delta_const     =-1
                delta_covered   =y_deltas[i] - (ypt - L_t)
                i   =i # don't change the section index!
                ypt =L_t
                
            elif ypt <= 0: # the ypt is below the receiver, need to switch directions
                is_at_panel_end =True
                delta_const     =1
                delta_covered   =y_deltas[i] - (0 - ypt)
                i   =i
                ypt =0
            else:
                i=i+1
            
 
        # populate the panel dictionary
        panel_dict['fluxes']    =panel_fluxes
        panel_dict['ypts']      =ypts

        # add panel to receiver dictionary
        xpt_name =f'{xpt:.2f}'
        rec_flux_dict[xpt_name] = panel_dict

        # move to the next panel
        xpt = xpt+W_panel
        xpts.append(xpt)

    return rec_flux_dict    # NOTE: this is only half of the receiver. Takes advantage of flowpath symmetry.

def plot_ideal_fluxmap(rec_flux_dict,H,W,flow_config,img_name = 'default_save'):
    """
    plots the ideal flux map as exact to geometric values as possible
    ---
    H - receiver height (m)
    W - receiver width (m)
    flow_config - string. Options: "2_ctr"
    """
    xpts        =np.array(list(rec_flux_dict.keys()),dtype=float)
    xpt_keys    =list(rec_flux_dict.keys())

    figDMG,axDMG      =plt.subplots(tight_layout=True)
    color_scheme = 'viridis'
    cscale    =matplotlib.pyplot.get_cmap(color_scheme, 100)
    # find the maximum flux
    flux_max =0
    flux_min =1e6
    for (xpt,xpt_key) in zip(xpts,xpt_keys):
        flux_array             =np.array(rec_flux_dict[xpt_key]['fluxes'])
        flux_max_challenger    =flux_array.max()
        valid_for_min          =np.array(flux_array > 1) # neglect zero flux locations

        flux_min_challenger    =flux_array[valid_for_min].min()

        flux_max = np.max([flux_max_challenger,flux_max])
        flux_min = np.min([flux_min_challenger,flux_min])
    ## loop through each panel (semelhante do que xpts, ne?)
    for (xpt,xpt_key) in zip(xpts,xpt_keys):
        x_corner=xpt
        W_section = np.diff(xpts)[0]
        y_heights_array = np.array(rec_flux_dict[xpt_key]['ypts'])
        # add start or end point to panel, depending on flowing with gravity or not
        if y_heights_array[0]==0:
            y_heights_array = np.concatenate( (y_heights_array, np.array([H])) )
        elif y_heights_array[0]==H:
            y_heights_array = np.concatenate( (y_heights_array, np.array([0]) ) )
        else:
            print('unexpected y start point encountered')
        y_deltas = np.diff(y_heights_array)
        fluxes   = np.array(rec_flux_dict[xpt_key]['fluxes'])
        for j, ypt in enumerate(y_heights_array[:-1]):
            y_corner=ypt
            fraction = fluxes[j]/flux_max #(fluxes[j]-flux_min)/(flux_max-flux_min)
            box_color= cscale(fraction)
            # print(fraction)
            axDMG.add_patch( patch.Rectangle((x_corner,y_corner), W_section, y_deltas[j], color=box_color ) )
            if flow_config == '2_ctr':
                axDMG.add_patch( patch.Rectangle((-x_corner,y_corner), -W_section, y_deltas[j], facecolor=box_color, edgecolor=box_color ) )
            else:
                print('flow configuration not recognized')
            # ax.add_patch( patch.Rectangle((x_corner,y_corner), W_section, y_deltas[j], edgecolor = box_color, facecolor='none' ) )
    
    ## make a color bar, then interpret every box color relative to that color bar
    axDMG.set_xlim(-W/2,W/2)
    axDMG.set_ylim(0,H)
    axDMG.set_xlabel('x-coordinate (m)',fontsize=12)
    axDMG.set_ylabel('y-coordinate (m)',fontsize=12)
    divider =make_axes_locatable(axDMG)                           # thanks, stack exchange!
    cax     =divider.append_axes("right", size="2%", pad=0.08)     # ^ thanks
    cb      =figDMG.colorbar(cm.ScalarMappable(matplotlib.colors.Normalize(vmin=0, vmax=flux_max/1e3,clip=False),cmap=color_scheme), cax=cax)
    cb.set_label('incident flux (kW/m$^2$)',  fontsize=12)
    axDMG.set_aspect('equal')
    plt.savefig(f'imgs/ideal_fluxmap_{img_name}.png',dpi=300)
    plt.show()
    plt.close(figDMG)
    ## plot each box, assign color relative to color bar

    return

def plot_ideal_fluxmap_w_offset(rec_flux_dict, H, W, flow_config, img_name = 'default_save'):
    """
    plots the ideal flux map as exact to geometric values as possible
    ---
    rec_flux_dict   - flux dictionary, with panels as keys and each panel contains ypts & corresponding lambda flux functions
    H - receiver height (m)
    W - total receiver width (m)
    flow_config - string. Options: "2_ctr"
    """
    xpts        =np.array(list(rec_flux_dict.keys()),dtype=float)
    xpt_keys    =list(rec_flux_dict.keys())

    figDMG,axDMG    =plt.subplots(tight_layout=True)
    color_scheme    ='viridis'
    cscale          =matplotlib.pyplot.get_cmap(color_scheme, 100)

    # find the maximum flux
    flux_max =0 # just an initial value
    # loop through every panel, create arrays of flux values
    for xpt_key in xpt_keys:
        flux_func_list          =np.array(rec_flux_dict[xpt_key]['fluxes'])
        # loop through every lambda function find maximum fluxes in each section
        for i,flux_func in enumerate(flux_func_list):
            ypts                    =rec_flux_dict[xpt_key]['ypts']
            ypt                     =ypts[i]
            flux_max_challenger     =flux_func(ypt)
            flux_max                =np.max([flux_max_challenger,flux_max])

    ## iterate through each panel
    for (xpt,xpt_key) in zip(xpts,xpt_keys):
        x_corner        =xpt
        W_section       =np.diff(xpts)[0]
        y_heights_array =np.array(rec_flux_dict[xpt_key]['ypts'])
        # add start or end point to panel, depending on flowing with gravity or not
        if y_heights_array[0]==0:
            y_heights_array =np.concatenate( (y_heights_array, np.array([H])) )
            is_flow_up      =True
        elif y_heights_array[0]==H:
            y_heights_array = np.concatenate( (y_heights_array, np.array([0]) ) )
            is_flow_up      =False
        else:
            print('unexpected y start point encountered')
        y_deltas = np.diff(y_heights_array)
        fluxes_func_list   = np.array(rec_flux_dict[xpt_key]['fluxes'])
        # iterate through every ypt, but have a subroutine for the offset region NOTE: unfinished as of 10/15/2025
        for j, ypt in enumerate(y_heights_array[:-1]):
            # case of starting the panel
            if j == 0:
                # generate a series of mini rectangles in starting offset region, each with a lambda-determined flux
                smidge          =-1e-6 if is_flow_up else 1e-6 # the lambda functions return "None"
                y_offset_pts    =np.linspace(y_heights_array[0], y_heights_array[1]+smidge, 200)
                y_offset_deltas =np.diff(y_offset_pts)
                offset_fractions=np.array([fluxes_func_list[0](pt) for pt in y_offset_pts])/flux_max
                offset_colors   =cscale(offset_fractions)
                for ii, y_sub_pt in enumerate(y_offset_pts[:-1]):
                    axDMG.add_patch(patch.Rectangle((x_corner, y_sub_pt), W_section, y_offset_deltas[ii], color=offset_colors[ii], linewidth=0))
                    if flow_config == '2_ctr':
                        axDMG.add_patch( patch.Rectangle((-x_corner,y_sub_pt), -W_section, y_offset_deltas[ii], color=offset_colors[ii], linewidth=0 ) )
            # finishing any panel except the final one
            elif (j == y_heights_array.size-2): #and ( xpt_key != xpt_keys[-1]):
                # same process as for j==0 case
                smidge          =-1e-6 if is_flow_up else 1e-6 # the lambda functions return "None"
                y_offset_pts    =np.linspace(y_heights_array[-2], y_heights_array[-1]+smidge, 200)
                y_offset_deltas =np.diff(y_offset_pts)
                offset_fractions=np.array([fluxes_func_list[-1](pt) for pt in y_offset_pts])/flux_max
                offset_colors   =cscale(offset_fractions)
                for ii, y_sub_pt in enumerate(y_offset_pts[:-1]):
                    axDMG.add_patch(patch.Rectangle((x_corner, y_sub_pt), W_section, y_offset_deltas[ii], color=offset_colors[ii], linewidth=0))
                    if flow_config == '2_ctr':
                        axDMG.add_patch( patch.Rectangle((-x_corner,y_sub_pt), -W_section, y_offset_deltas[ii], color=offset_colors[ii], linewidth=0) )
            # all other cases. ie in the middle sections of the panel
            else:
                y_corner=ypt
                fraction = fluxes_func_list[j](ypt)/flux_max #(fluxes[j]-flux_min)/(flux_max-flux_min)
                box_color= cscale(fraction)
                axDMG.add_patch( patch.Rectangle((x_corner,y_corner), W_section, y_deltas[j], color=box_color, linewidth=0 ) )
                if flow_config == '2_ctr':
                    axDMG.add_patch( patch.Rectangle((-x_corner,y_corner), -W_section, y_deltas[j], color=box_color, linewidth=0 ) )
                else:
                    print('flow configuration not recognized')
                # ax.add_patch( patch.Rectangle((x_corner,y_corner), W_section, y_deltas[j], edgecolor = box_color, facecolor='none' ) )
    
    ## make a color bar, then interpret every box color relative to that color bar
    axDMG.set_xlim(-W/2,W/2)
    axDMG.set_ylim(0,H)
    axDMG.set_xlabel('x-coordinate (m)',fontsize=12)
    axDMG.set_ylabel('y-coordinate (m)',fontsize=12)
    divider = make_axes_locatable(axDMG)                           # thanks, stack exchange!
    cax = divider.append_axes("right", size="2%", pad=0.08)     # ^ thanks
    cb = figDMG.colorbar(cm.ScalarMappable(matplotlib.colors.Normalize(vmin=0, vmax=flux_max/1e3,clip=False),cmap=color_scheme), cax=cax)
    cb.set_label('incident flux (kW/m$^2$)',  fontsize=12)
    axDMG.set_aspect('equal')
    plt.savefig(f'imgs/ideal_fluxmap_w_offset_{img_name}.png',dpi=300)
    plt.show()
    plt.close(figDMG)

    return

def plot_ideal_fluxgrid(fluxgrid, H, W, flow_config,img_name='default_save'):
    """
    plots the fluxgrid realistically by scaling grid to the height and width
    ---
    fluxgrid    - (W/m2) 2d array
    H           - (m) height
    W           - (m) width 
    flow_config - (str) valid options currently include: 2_ctr
    img_name    - (str) desired png save name
    """
    res_y =fluxgrid.shape[0]  # resolution, assuming a square array
    res_x =fluxgrid.shape[1]
    if flow_config == '2_ctr':
        fontsize=14
        W_half =W/2                 
        xgrid_pts =np.linspace(-W_half,W_half-(W/res_x),res_x)+0.5*W/res_x
        ygrid_pts =np.linspace(0,H-(H/res_y),res_y)+0.5*H/res_y
        fig,ax =plt.subplots(tight_layout=True)
        ax.set_xlabel('x coordinate (m)',fontsize=fontsize)
        ax.set_ylabel('y coordinate (m)',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        mapa =ax.imshow(fluxgrid/1000, extent =[-W/2,W/2,0,H])
        divider = make_axes_locatable(ax)  
        cax = divider.append_axes("right", size="2%", pad=0.08)     # thanks stack exchange!
        cb =fig.colorbar(mapa,cax=cax)
        cb.set_label('incident flux (kW/m2)',fontsize=fontsize)
        # # cb = fig.colorbar(cm.ScalarMappable(matplotlib.colors.Normalize(vmin=0, vmax=flux_max/1e3,clip=False),cmap=color_scheme), cax=cax)
        # ## behold: a lazy man's meshgrid
        # xlist =[]
        # ylist =[]
        # for x in xgrid_pts:
        #     for y in ygrid_pts:
        #         xlist.append(x)
        #         ylist.append(y)
        #         # ax.scatter(x,y,color='k',marker='.',s=1)
        #         # print(x,y)
        # ax.scatter(xlist,ylist,color='k',marker='.',s=1)
        # ax.set_xlabel('x position (m)', fontsize=14)
        # ax.set_ylabel('y position (m)', fontsize=14)
        # ##
        plt.savefig(f'imgs/{img_name}_flux_grid_map.png',dpi=300)
        plt.show()
    else:
        print('flow configuration not recognized')
    return

def save_ideal_fluxmap(rec_dict,filestring):
    """
    saves rec_dict in a json for later use. Flux is in (W/m2)
    ---
    rec_dict:   dictionary of panel dictionaries
    filestring: name and filepath for json save

    """
    with open(filestring,"w") as f:
        json.dump(rec_dict,f)

    return

def fully_utilize_receiver_maximize_lifetime(dmg_case, tube, mflow, W_panel, W_receiver, L_t, v_offset, flowpath_config,cutoff=270):
    """
    assumes a constant image offset and finds the maximum lifetime contour that can fit on the receiver. 
    ---
    dmg_case -  damage tool object instance
    tube     -  initialized receiver model tube
    mflow    -  total system mass flow
    W_panel  -  panel width
    L_t      -  tube length or receiver height
    v_offset -  image offset in the vertical direction
    flowpath_config - flowpath configuration. Only current option is '2_ctr'
    --- returns ---
    rec_flux_dict - (dict) receiver dictionary. If offset=0
                then each panel's flux array is a series of values. If offset >0
                then the flux array is a series of flux functions
    Q_inc_rec     - (W) incident thermal receiver power
    Areas         - area of ONE SIDE of required areas corresponding to each flux value/lambda function
    A_req         - total area required
    """
    # determine if the offset is nonzero
    is_offset_zero          =True if v_offset ==0 else False
    is_correct_LTE_decade   =False
    is_correct_LTE          =False

    # iterate through lifetime levels until we find the highest possible lifetime
    LTE_incr    =80 # (yrs) starting increment
    LTE_des     =30 # (yrs) initial desired lifetime
    Wreq        =1  # (m) initial required TOTAL width to get loop started
    iter        =0
    while not is_correct_LTE:

        # if non zero, then implement the fluxmap generater that considers offset
        if is_offset_zero:
            rec_dict, Q_inc_rec, Areas, A_reqd   =generate_ideal_fluxmap(dmg_case, tube, LTE_des, mflow, W_panel, L_t, flowpath_config, cutoff=cutoff)
        # if zero, then implement the old fluxmap 
        else:
            rec_dict, Q_inc_rec, Areas, A_reqd   =generate_ideal_fluxmap_with_offset(dmg_case, tube, LTE_des, mflow, W_panel, L_t, v_offset, flowpath_config, show_LTE_ctr=False, cutoff=cutoff)

        # compare the width to the allowable width
        if flowpath_config == '2_ctr':
            nPanels_1side   =len(list(rec_dict.keys()))
            Wreq            =W_panel*nPanels_1side*2
        else:
            print('flowpath not recognized!')

        if LTE_incr <= 5:
            is_correct_LTE =True
        # if width is less, increment by (step) years. use a is_fully_utilized boolean
        elif (Wreq < W_receiver) and not is_correct_LTE_decade:
            LTE_des =LTE_des+LTE_incr
        # if width is more, decrement by (step/2) years.
        elif (Wreq > W_receiver) and not is_correct_LTE_decade:
            LTE_incr    =LTE_incr/2
            LTE_des     =LTE_des-LTE_incr
        elif (Wreq == W_receiver) and not is_correct_LTE_decade:
            print(f'--- {Wreq/W_panel} panels fit on the {W_receiver} wide receiver ---')
            is_correct_LTE_decade =True
        # if we are in the correct decade, slowly increase the LTE until we add another panel
        elif is_correct_LTE_decade and (Wreq == W_receiver):
            LTE_des =LTE_des+LTE_incr
        # we were in the correct decade until we added too many LTE years; decrement
        elif is_correct_LTE_decade and (Wreq >= W_receiver):
            LTE_des =LTE_des-LTE_incr
            LTE_incr=LTE_incr/2

        
        if LTE_des < 0:
            print('your receiver sucks! change the dimensions.')
            quit()
        
        iter+=1

    ## output should use the decided LTE  to generate final fluxmap
        # if non zero, then implement the fluxmap generater that considers offset
    if is_offset_zero:
        rec_dict, Q_inc_rec, Areas, A_reqd   =generate_ideal_fluxmap(dmg_case, tube, LTE_des, mflow, W_panel, L_t, flowpath_config, cutoff=cutoff)
    # if zero, then implement the old fluxmap 
    else:
        rec_dict, Q_inc_rec, Areas, A_reqd   =generate_ideal_fluxmap_with_offset(dmg_case, tube, LTE_des, mflow, W_panel, L_t, v_offset, flowpath_config, show_LTE_ctr=True, cutoff=cutoff)
    ##
    print('\n')
    print(f'--- number of iterations:{iter} ---')
    print(f'--- final selected lifetime: {LTE_des} years ---')
    print('\n')
    return rec_dict, Q_inc_rec, Areas, A_reqd

def build_ideal_fluxgrid(filestring, res_y, H, W, flowpath_config):
    """
    loads ideal fluxgrid json file and delivers a flux grid at the desired resolution
    --
    filestring - (str) name and path of ideal flux map json file to load. json should only be one receiver half for 2_ctr config.
    res        - (int) desired flux grid resolution in y dimension
    flowpath_config- (str) flowpath configuration. valid options are currently: '2_ctr'
    H          - (m) receiver height
    W          - (m) full receiver width
    ---
    returns:    flux_grid  - res x res np.array. each grid is in the centroid of its respective square
                                note: in row major, starting with y=H. So flux_grid[0][0] is @x=0,y=H. flux_grid[-1][-1] is @x=L,y=0
    """
    ## load ideal fluxgrid
    with open (filestring,) as f:
        dict=json.load(f)

    ## extract the panel keys from dictionary
    xpts        =np.array(list(dict.keys()),dtype=float)    # the xpts directly depend on the number of panels
    panel_keys  =list(dict.keys())                          # get a list of panel names

    
    if flowpath_config == '2_ctr':
        res_x   =len(panel_keys)*2                          # the receiver dictionary for 2_ctr only has one half of the receiver's fluxes
        res_y   =res_y                                      
        flux_grid =np.ones((res_y,res_x))                   # pre-allocate flux_grid  
        ## construct x,y grid points.
        W_half =W/2
        # construct a grid with each point at a centroid, just like Solarpilot
        xgrid_pts =np.linspace(-W_half,W_half-(W/res_x),res_x)+0.5*W/res_x 
        ygrid_pts =np.linspace(0,H-(H/res_y),res_y)+0.5*H/res_y
        # want to go in row major order
        ygrid_pts =ygrid_pts[::-1]
        for i,y in enumerate(ygrid_pts):
            for j,x in enumerate(xgrid_pts):
                panel_key_index =np.where(xpts == xpts[abs(x) > xpts].max())[0][0]  # compare x to xpts, pick the panel with the nearest x position to the left of x
                panel_key       =panel_keys[panel_key_index]                        
                panel           =dict[str(panel_key)]

                ypts = np.array(panel['ypts'])  # compare y to ypts, find closest upstream ypt to index the correct flux
                if ypts[0]==0:  # this is the ascending case. In this case we need to find the greatest ypt that y is greater than
                    flux_index =np.where(ypts ==ypts[abs(y) >ypts].max())
                else:           # this is the descending case. Need to find the least point that y is less than
                    flux_index =np.where(ypts ==ypts[abs(y) < ypts].min())
                flux_index =flux_index[0][0]    # two indexes because np.where is weird...
                flux_pt   =panel['fluxes'][flux_index]
                flux_grid[i,j] =flux_pt
    else:
        print('flowpath configuration not recognized')

    return flux_grid

def build_ideal_w_offset_fluxgrid(dict, res_y, H, W, flowpath_config):
    """
    loads ideal fluxgrid json file and delivers a flux grid at the desired resolution
    --
    dict       - (str) receiver dictionary
    res        - (int) desired flux grid resolution - function will make a square
    flowpath_config- (str) flowpath configuration. valid options are currently: '2_ctr'
    ---
    returns:    flux_grid  - res x res np.array. each grid is in the centroid of its respective square
                                note: in row major, starting with y=H. So flux_grid[0][0] is @x=0,y=H. flux_grid[-1][-1] is @x=L,y=0
                H          - (m) receiver height
                W          - (m) full receiver width

    """
    ## extract the panel keys from dictionary
    xpts        =np.array(list(dict.keys()),dtype=float)    # the xpts directly depend on the number of panels
    panel_keys  =list(dict.keys())                          # get a list of panel names

    
    if flowpath_config == '2_ctr':
        res_x   =len(panel_keys)*2                          # the receiver dictionary for 2_ctr only has one half of the receiver's fluxes
        res_y   =res_y                                      
        flux_grid =np.ones((res_y,res_x))                   # pre-allocate flux_grid  
        ## construct x,y grid points.
        W_half =W/2
        # construct a grid with each point at a centroid, just like Solarpilot
        xgrid_pts =np.linspace(-W_half,W_half-(W/res_x),res_x)+0.5*W/res_x 
        ygrid_pts =np.linspace(0,H-(H/res_y),res_y)+0.5*H/res_y
        # want to go in row major order
        ygrid_pts =ygrid_pts[::-1]
        for i,y in enumerate(ygrid_pts):
            for j,x in enumerate(xgrid_pts):
                panel_key_index =np.where(xpts == xpts[abs(x) > xpts].max())[0][0]  # compare x to xpts, pick the panel with the nearest x position to the left of x
                panel_key       =panel_keys[panel_key_index]                        
                panel           =dict[str(panel_key)]

                ypts = np.array(panel['ypts'])  # compare y to ypts, find closest upstream ypt to index the correct flux
                if ypts[0]==0:  # this is the ascending case. In this case we need to find the greatest ypt that y is greater than
                    flux_index =np.where(ypts ==ypts[abs(y) >ypts].max())
                else:           # this is the descending case. Need to find the least point that y is less than
                    flux_index =np.where(ypts ==ypts[abs(y) < ypts].min())
                flux_index =flux_index[0][0]    # two indexes because np.where is weird...
                flux_func =panel['fluxes'][flux_index]
                flux_pt   =flux_func(y)
                flux_grid[i,j] =flux_pt
    else:
        print('flowpath configuration not recognized')

    return flux_grid

def find_critical_resolution(rec_dict_file_string, Qrec, H, W, flowpath_config):
    """
    Identifies the critical resolution required for a flux grid to accurately depict the flux map
    ---
    rec_dict_file_string    - (str) json string for receiver dictionary, which contains panels with x coordinates as keys.
                                ...Each panel contains an array of y points immediately upstream from their respective fluxes
    Qrec                    - (float) the total integrated reciever power of the rec_dict input
    H                       - (m) receiver height
    W                       - (m) receiver total width
    flowpath_config         - (str) flowpath configuration
    """
    Q_inc_rec_grids =[]
    ress=np.arange(10,310,10)
    for res in ress: 
        flux_grid       =build_ideal_fluxgrid(rec_dict_file_string,res,H,W,flowpath_config)
        flux_avg        =np.mean(flux_grid)
        Q_inc_rec_grid  =flux_avg*H*W
        Q_inc_rec_grids.append(Q_inc_rec_grid/1e6)
    
    fig,ax  =plt.subplots()
    ax.scatter(ress, Q_inc_rec_grids,label='fluxgrid',s=5)
    ax.hlines(Qrec/1e6, ress.min(), ress.max(), linestyles='--', colors='k',label='fluxmap')
    ax.grid(True, alpha=0.5)
    ax.set_ylabel('total incident power (MW$_{th}$)', fontsize=12)
    ax.set_xlabel('y resolution', fontsize=14)
    ax.set_xlim( ress.min(), ress.max() )
    plt.savefig('imgs/resolution_plot_for_fluxgrid_default_save',dpi=300)
    plt.show()
    plt.close()

    return

def fit_grid_to_receiver(flux_grid_ideal, receiver, flowpath_config):
    """
    reconciles the ideal flux map, which uses as few panels as possible, to a desired receiver by adding grid points 
    ---
    flux_grid_ideal - npanel x nz array of flux points assuming an ideal map
    Npanels_ideal   - (int) number of required panels
    receiver        - (object) the receiver thermal model
    flowpath_configuration  - (str)
    """
    ## shape of the flux grid tells you number of panels, because we assume that x resolution is Npanels
    Npanels_ideal   =flux_grid_ideal.shape[1]
    nz              =flux_grid_ideal.shape[0]

    ## get the actual number of panels
    Npanels_actual  =receiver.Npanels

    ## calculate the number of panels to add
    delta_panels    =Npanels_actual-Npanels_ideal
    if delta_panels < 0:
        print('the receiver is too small')
        quit()
    else:
        ## tack on a column before and after the flux_grid_ideal to account
        if flowpath_config == '2_ctr':
            nAdd            =int(delta_panels/2)
            flux_grid_appended =np.concatenate((np.zeros((nz,nAdd)),flux_grid_ideal, np.zeros((nz,nAdd))),axis=1)
    

    return flux_grid_appended

if __name__ == "__main__":
    ## testing the receiver builder function
    x_start =0
    x_end =1.5
    y_start =1
    y_deltas = [0.7, 0.8, 0.7, 0.8]
    W_panel =1
    L_t =1
    fluxes_pred =[100, 300, 50, 1000]
    rec_dict =assign_fluxes_to_panels(x_start, y_start, y_deltas, W_panel, L_t, fluxes_pred)
     
    ## test getting the damage tool
    dmg_model = damage_tool.damageTool('A230')
    dmg_model.make_contour_function(30)

    # instantiate the tube, check if HTC is reasonable for a test value
    ## --- these lines needed in main
    tube =tube_jwenner.Tube()
    tube.OD =0.0508             # (m)
    tube.twall =0.00125         # (m)
    tube.initialize()
    ## ---
    ntubes_panel =int(W_panel/tube.OD)
    mflow_total =250             # (kg/s)
    npaths =2
    mflow_tube =mflow_total/npaths/ntubes_panel
    tube.mflow =mflow_tube
    h, Re, Vel = tube.internal_h(Tf =565+273.15)


    ## test the entire shabang
    tube            =tube_jwenner.Tube()
    tube.OD         =0.0508             # (m)
    tube.twall      =0.00125         # (m)
    tube.initialize()
    W_panel=1
    L_t=10
    flowpath_config='2_ctr'
    mflow=200
    LTE_goal=30
    

    # ## test original methods
    # rec_dict, Q_inc_rec, Areas, Areqd =generate_ideal_fluxmap(damage_tool.damageTool('A230'), tube, LTE_goal, mflow, W_panel, L_t, flowpath_config)

    # W=W_panel*len(rec_dict.keys())*2
    
    # plot_ideal_fluxmap(rec_dict,L_t,W,'2_ctr','demo_case')
    # save_ideal_fluxmap(rec_dict, 'test_json.json')
    # flux_grid =build_ideal_fluxgrid('test_json.json', res_y=50, H=L_t, W=W, flowpath_config='2_ctr')
    # plot_ideal_fluxgrid(flux_grid, L_t, W, '2_ctr','test_w_no_offset')
    # # find_critical_resolution('test_json.json', Q_inc_rec, L_t, W, '2_ctr')
    # ##

    # new way to generate an ideal flux map
    v_offset    =0.5 # need high res if offset region is small
    
    rec_dict, Q_inc_rec, Areas, Areqd   =generate_ideal_fluxmap_with_offset(damage_tool.damageTool('A230'), tube, LTE_goal, mflow, W_panel, L_t, v_offset, flowpath_config)    
    # rec_dict, Q_inc_rec, Areas, Areqd   =generate_ideal_fluxmap(damage_tool.damageTool('A230'), tube, LTE_goal, mflow, W_panel, L_t, flowpath_config)    
    nPanels     =len(list(rec_dict.keys()))
    W           =W_panel*len(rec_dict.keys())*2 + 2
    W_receiver  =W
    plot_ideal_fluxmap_w_offset(rec_dict, L_t, W, flowpath_config)
    
    # test the utilize_receiver function
    rec_dict, Q_inc_rec, Areas, Areqd   =fully_utilize_receiver_maximize_lifetime(damage_tool.damageTool('A230'), tube, mflow, W_panel, W_receiver, L_t, v_offset, flowpath_config)
    plot_ideal_fluxmap_w_offset(rec_dict, L_t, W, flowpath_config)
    
    fluxgrid                            =build_ideal_w_offset_fluxgrid(rec_dict, res_y=100, H=L_t, W=W, flowpath_config=flowpath_config)
    plot_ideal_fluxgrid(fluxgrid, L_t, W, flowpath_config, img_name=f'test_w_offset_{v_offset}m')
    print('congrats, name == main')