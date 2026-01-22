"""
created by jwenner on 8/13/2025 to develop and test a 3D heuristic for placing images in the least damaging arrangement possible
"""
## append the native HALOS folder to path for importing
import sys
sys.path.append('./../HALOS_code/')
sys.path.append('./../../Chapter_3/thermal_and_optical_tools/')
sys.path.append('./../aiming_informer/')
sys.path.append('./../HALOS_code/')
sys.path.append('./../../Chapter_3/damage_tool/')
import helpers_thermal_model as therm_helpers 
import pandas as pd
import numpy as np
import matplotlib.animation as animation
import timeit
import informed_aiming
import inputs
import matplotlib.pyplot as plt
import sp_module
import damage_tool
import tube_jwenner
import helpers_heuristic
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import json
import multiprocessing
import copy

def flux_to_LTE_w_MP(kwarg):
    """
    single function that invokes all the steps necessary to estimate the panel lifetimes
    ---
    kwarg         - dictionary that includes the following:
        thermal_clone - independent thermal model object 
        dmg_clone     - deep copy of damage model
        flux_option         - (kW/m2) flux profile that needs to be tested
    returns
    LTE_avg - average panel lifetime based on minimum tube lifetimes
    """
    thermal_clone       =kwarg['thermal_clone']      
    dmg_clone           =kwarg['dmg_clone']
    flux_option         =kwarg['flux_profile_option']
    RMSE_standard       =kwarg['RMSE_std']
    
    therm_helpers.solve_LWT_thermal_model(thermal_clone,flux_option)
    dTs, Tfs, qabs, Rs                              =therm_helpers.get_thermal_results(thermal_clone)
    LTEs                                            =dmg_clone.get_LTEs_w_penalties(dTs.flatten(),Tfs.flatten(),Rs.flatten())
    min_panel_LTEs_alt_opt, min_tube_LTEs_alt_opt   =dmg_clone.calc_minimum_panel_LTEs(thermal_clone, LTEs) # required for mean and min RMSE methods
    min_panel_LTEs_alt_opt                          =np.clip(min_panel_LTEs_alt_opt, -10000, RMSE_standard) # only give low LTEs a vote
    LTE_avg                                         =np.average(min_panel_LTEs_alt_opt)  # mean panel lifetime method

    return LTE_avg

if __name__ =='__main__': # very important if multiprocessing. Prevents subprocesses from being started

    ## make receiver thermal models and unpack related inputs
    ### setup all required thermal models
    receiver_filestring ='./../receivers/heuristic_receiver'
    model_lite          =therm_helpers.setup_LWT_thermal_model(filestring=receiver_filestring,ntubes_sim=1)
    model_hvy           =therm_helpers.setup_LWT_thermal_model(filestring=receiver_filestring,ntubes_sim=3)

    ## access / create receiver files
    ## set up HALOS object for characterization use
    casename        ="compare_3D_heuristic_140"
    case_filename   =f"./../case_inputs/{casename}.csv"
    filenames       =inputs.readCaseFile(case_filename)
    use_sp_field    =False

    flux_instance   =sp_module.SP_Flux(filenames, use_sp_field=use_sp_field)
    HALOS_res   =model_lite.Npanels*3   # the square resolution for solarpilot simulation, ideal flux grid, and thermal model input
    flux_instance.receiver_data['pts_per_dim']  =HALOS_res

    if os.path.exists(f"./../case_inputs/{casename}/{casename}.json"):
        with open (f"./../case_inputs/{casename}/{casename}.json",) as f:
            flux_dict=json.load(f)
    else:
        flux_dict       =flux_instance.get_single_helio_flux_dict(aim_method='Simple aiming method')
        with open(f"./../case_inputs/{casename}/{casename}.json","w") as f:
            json.dump(flux_dict,f)

    W           =float(flux_instance.receiver_data['length'])
    H           =float(flux_instance.receiver_data['height'])
    A           =H*W

    ## iterate row by row and get the image widths, heights, and powers
    threshhold=0.01
    image_widths    ={}
    image_heights   ={}
    width_list      =[]
    height_list     =[]
    image_rel_areas =np.array([])
    # image_powers=np.array([])
    # image_peak_fluxes=np.array([])
    for key in flux_dict.keys():
        flux_arr    =np.array(flux_dict[key])
        # for row in flux_dict[key]:
        #     flattened_profile=np.maximum(flattened_profile, np.array(row))
        # calculate the fractional dimensions by dividing number of occupied grid pts to total pts
        fractional_width =np.sum(flux_arr.max(axis=0) > threshhold*flux_arr.max())/HALOS_res 
        fractional_height=np.sum(flux_arr.max(axis=1) > threshhold*flux_arr.max())/HALOS_res

        # add the image height and widths to the master dictionaries and lists
        image_widths[key]=W*fractional_width
        image_heights[key]=H*fractional_height
        width_list.append(image_widths[key])
        height_list.append(image_heights[key])

        ## calculate image power
        flux_avg=np.average(flux_arr)
        image_power=flux_avg*A

        # image_powers = np.concatenate( (image_powers, np.array([image_power]) ) )
        ## get peak flux of image
        # image_peak_flux=np.max(flattened_profile)
        # image_peak_fluxes = np.concatenate( (image_peak_fluxes, np.array([image_peak_flux]) ) )

        ## calculate total image area
        npts_w_flux     =np.sum( flux_arr > threshhold*flux_arr.max() )
        image_area_rel  =npts_w_flux/(HALOS_res*HALOS_res) 
        image_area      =image_area_rel*A if image_area_rel <= 1 else A
        image_rel_areas =np.concatenate( ( image_rel_areas, np.array([image_area_rel]) ) )

    n_images        =image_rel_areas.size  # get the number of images for record keeping
    print(f'total number of heliostats is: {n_images}')
    asc_image_list  =np.argsort(image_rel_areas) # create an image list in ascending order
    
    ## make a dataframe with all widths, areas, heights, and keys
    image_dict              ={}
    image_dict['widths']    =width_list
    image_dict['heights']   =height_list
    image_dict['rel_areas'] =image_rel_areas.tolist()
    image_dict['keys']      =list(flux_dict.keys())
    image_df                =pd.DataFrame(dict([(key, pd.Series(value)) for key, value in image_dict.items()]))
    image_df                =image_df.set_index('keys')
    image_df                =image_df.sort_values('widths',ascending=False)

    ncut                    =0
    # remove first ncut heliostats
    image_df                =image_df.iloc[ncut:]
    ## instantiate a damage case
    mat_string ='A230'
    dmg_inst =damage_tool.damageTool(mat_string)   

    ## generate ideal flux maps for slack flux comparison
    tube        =tube_jwenner.Tube()
    tube.OD     =0.0508             # (m)
    tube.twall  =0.00125         # (m)
    tube.options.is_adjacent_tubes =False # prevents annoying error when initializing the tube
    tube.initialize()

    LTE_desired     =80
    LTE_trigger     =240 # this is the trigger to start optimizing according to LTE impact
    LTE_threshold   =15
    LTE_rmse        =LTE_trigger
    mflow           =np.sum(model_lite.operating_conditions.mass_flow)  # the setup_LWT_thermal_model automatically selects a mass flow rate based on the design power and outlet temp. range
    W_panel         =model_lite.D/model_lite.Npanels # for now, the width is determined by the number of panels, which is set so that the velocity matches the reference velocity of the prelim study
    L_t             =model_lite.tubes[0][0].length
    flowpath_config =f'{model_lite.npaths}_{model_lite.start_pt}'
    rec_flux_dict, Q_inc_rec, Areas, A_req =informed_aiming.generate_ideal_fluxmap(dmg_inst, tube, LTE_desired, mflow, W_panel, L_t, flowpath_config)

    # plot and save json of what the ideal fluxmap looks like
    filestring_fluxmap =f"./../case_inputs/{casename}/{casename}_3D_heuristic_dev_fluxmap.json"
    # informed_aiming.plot_ideal_fluxmap(rec_flux_dict, L_t, model_lite.D, '2_ctr', casename)
    informed_aiming.save_ideal_fluxmap(rec_flux_dict, filestring_fluxmap)

    # translate the flux map into a flux grid NOTE: not a square grid unless res_y set to Npanels to match res_x
    # also note that the ideal flux grid is in units of W

    # need to check the width and raise error if the ideal fluxmap requires more panels than the receiver has
    if flowpath_config == '2_ctr':
        N_panels_half   =len(rec_flux_dict.keys())
        N_panels_ideal  =N_panels_half*2

    
    ideal_flux_grid =informed_aiming.build_ideal_fluxgrid(filestring_fluxmap,res_y=model_lite.Npanels, H=L_t, W=W_panel*N_panels_ideal, flowpath_config=flowpath_config)
   
    if N_panels_ideal   !=np.round(model_lite.D/W_panel):
        print('Error! The minimum panel number for receiver @ this height differs from that in receiver object. adapting to match thermal model')
        ideal_flux_grid =informed_aiming.fit_grid_to_receiver(ideal_flux_grid, model_lite, flowpath_config)
    
    informed_aiming.plot_ideal_fluxgrid(ideal_flux_grid, L_t, model_lite.D, flowpath_config, casename)
    ideal_flux_grid =therm_helpers.increase_flux_resolution_blocked(flux_low_res = ideal_flux_grid, ndim_new = HALOS_res)

    # save the flux grid as a csv if desiring to use in SPT method
    aiming_df   =pd.DataFrame(data=ideal_flux_grid, index=None, columns=None, dtype=None, copy=None) # create empty dataframe 
    aiming_df   =aiming_df.iloc[::-1]    # flip this to prepare for SPT input
    aiming_df.to_csv(f'aiming/ideal_fluxgrid_{model_lite.Qdes}_LTE{LTE_desired}_for_SPT.csv',header=None, index=None)

    # pre-allocate flux profile array
    flux_profile    =np.zeros(ideal_flux_grid.shape)

    # initialize model solve for deep copies
    dummy_uniform_flux_grid =therm_helpers.increase_flux_resolution_blocked(flux_low_res = 200*np.ones(flux_profile.shape), ndim_new = HALOS_res)
    therm_helpers.solve_LWT_thermal_model(model_lite,dummy_uniform_flux_grid)
    _, Tfs_dummy, _, _              =therm_helpers.get_thermal_results(model_lite)

    ## this 'if nest' is a model check and initialization
    if not model_lite.tubes[0][0].flow_against_gravity: # assuming symmetry. If the left most panel is flowing w gravity then both outlet temps are at bottom of first/last array column
        T_out_left_dummy  =Tfs_dummy[Tfs_dummy.shape[0]-1,0]
        T_out_right_dummy =Tfs_dummy[Tfs_dummy.shape[0]-1,-1]
    else: # assuming symmetry. If the left most panel is against gravity then both outlet temps are at top of first/last array column
        T_out_left_dummy  =Tfs_dummy[0,0]
        T_out_right_dummy =Tfs_dummy[0,-1]
    print(f'initialized fluid temperature outlet temperatures are: {T_out_left_dummy} & {T_out_right_dummy}')

    ## define the x and y coordinates relative to the regulary spaced grid pattern
    if flowpath_config == '2_ctr':
        W_half =W/2
        res_x =res_y =int(flux_instance.receiver_data['pts_per_dim'])
        # construct a grid with each point at a centroid, just like Solarpilot
        xgrid_pts =np.linspace(-W_half,W_half-(W/res_x),res_x)+0.5*W/res_x 
        ygrid_pts =(np.linspace(0,H-(H/res_y),res_y)+0.5*H/res_y)[::-1]

    ## loop: placing images until the outlet temperature is T_out_obj [C]
    # dataframe has been sorted by area
    t_loop_start        =timeit.default_timer()
    f_og                =1 # how many sigmas to offset from edge
    is_left_done        =True   # tells the subselection to include or exclude the left or right sides
    is_right_done       =False
    T_out_obj           =565
    in_loop             =True   # initial value to allow the procedure to enter the while loop
    is_flux_violation   =False  # ititial value
    endgame             =False  # mode that only allows for placement on the edges of the receiver
    miter               =50
    placements          =[]


    ####----------------------- start of loop ------------------------------------
    for i in range(len(image_df.index)):
        in_loop     =True # refresh re-entry criteria every time i changes
        endgame     =False
        iteration   =0
        fx          =fy =f_og
        while in_loop ==True:
            iteration+=1
            ## print a progress update every 100 iterations
            if (i % 100) == 0:
                print(f'{i} images placed')

                # # get  a snapshot of the images placed so far
                # cbr_fontsize =12
                # fig,ax =plt.subplots()
                # im =ax.imshow(flux_profile/1e3, extent =[-model_lite.D/2,model_lite.D/2,0,model_lite.H])
                # ax.set_xlabel('x location (m)')
                # ax.set_ylabel('y location (m)')
                # divider = make_axes_locatable(ax)  
                # cax = divider.append_axes("right", size="2%", pad=0.08)     # thanks stack exchange!
                # cbr = fig.colorbar(im, cax=cax)
                # cbr.set_label(label='incident flux (kW/m$^2$)', size=cbr_fontsize)
                # cbr.ax.tick_params(labelsize=cbr_fontsize)
                # fig.savefig(f'imgs/default_final_flux_profile_{i}_images',dpi=300)
                # plt.close()

            # grab an image and all necessary info
            key             =image_df.index[i]
            flux_arr_og_kW  =np.array(flux_dict[key])
            img_width       =image_df.widths[key]
            img_height      =image_df.heights[key]

            ## get the location of the slack flux
            slack_profile   =ideal_flux_grid - flux_profile

            ## determine feasible aimpoints based on given offset and image's standard deviation
            sigma_x =img_width/6    # assuming that image width is roughly analogous to 6 standard deviations of the image
            sigma_y =img_height/6   # assuming that height behaves similarly to width
            if flowpath_config =='2_ctr':
                ctr_left        =np.where(xgrid_pts < 0)[0][-1]     # the node just to the left of center

                x_sub_indices   =np.where((xgrid_pts > ( (-W/2)+fx*sigma_x)) & (xgrid_pts < ( (W/2)-fx*sigma_x)))[0]
                x_ind_left      =x_sub_indices[0]   if is_left_done     == False else int(ctr_left+1)
                x_ind_right     =x_sub_indices[-1]  if is_right_done    == False else int(ctr_left)

                y_sub_indices   =np.where( (ygrid_pts < (H-fy*sigma_y)) & (ygrid_pts > (0+fy*sigma_y)) )[0]
                y_ind_bottom    =y_sub_indices[-1]
                y_ind_top       =y_sub_indices[0]

                if endgame: # the only x indices allowed in endgame mode is the right and left panel edges
                    x_ind_left  =x_sub_indices[-1] if is_right_done == False else x_sub_indices[0]
                    x_ind_right =x_ind_left
            slack_profile_subselection  =slack_profile[y_ind_top:y_ind_bottom+1, x_ind_left:x_ind_right+1]


            sub_grid_pt         =np.unravel_index(np.argmax(slack_profile_subselection),slack_profile_subselection.shape)
            # LTEs_options        =np.zeros(slack_profile_subselection.shape)  # initiate optional LTEs matrix for loop function
            ## define the desired grid point, but be sure to translate back into the original grid context
            y_move  =y_ind_top+sub_grid_pt[0]
            x_move  =x_ind_left+sub_grid_pt[1]
            # flux_arr_moved_kW  =helpers_heuristic.move_2d_array_from_ctr(flux_arr_og_kW, x_move, y_move)

            # # update the receiver flux profile with the new flux
            # flux_profile_option_W    =flux_profile + flux_arr_moved_kW*1e3
            # flux_profile_option_kW   =flux_profile_option_W/1e3


            ## if violation encountered, search other grid options
            if is_flux_violation:
                print(f'Minimum lifetime is {np.min(min_panel_LTEs_checkin):.0f}. Analyzing {slack_profile_subselection.size} options for image {key}')
                #
                # LTEs_options    =np.zeros(slack_profile_subselection.shape)
                increment               =2
                ## for multiprocessing only
                iterable                =[] # this needs to be a list of dictionaries
                # for single processing only
                # LTEs_options=[]
                # for multi and single processing
                coord_list              =[]
                ###
                for i_y in range(0,slack_profile_subselection.shape[0],increment):
                    for i_x in range(0,slack_profile_subselection.shape[1],increment):
                        #### vvv this is for running without multiprocessing
                #         #again, don't forget to put the x,y grid points in context of the total grid!
                #         y_move_opt  =y_ind_top+i_y
                #         x_move_opt  =x_ind_left+i_x
                #         flux_arr_moved_kW  =helpers_heuristic.move_2d_array_from_ctr(flux_arr_og_kW, x_move_opt, y_move_opt)
                #         # update the receiver flux profile with the new flux
                #         flux_profile_option_W    =flux_profile + flux_arr_moved_kW*1e3
                #         flux_profile_option_kW   =flux_profile_option_W/1e3
                #         ## solve lightweight thermal model
                #         flux_profile_option_kW          =therm_helpers.increase_flux_resolution_blocked(flux_low_res = flux_profile_option_kW, ndim_new = HALOS_res)
                #         # model_inst =copy.deepcopy(model_lite)   # model object edits
                #         therm_helpers.solve_LWT_thermal_model(model_lite,flux_profile_option_kW)
                #         dTs, Tfs, qabs, Rs              =therm_helpers.get_thermal_results(model_lite)
                #  #       LTEs                            =dmg_inst.get_LTEs(dTs.flatten(),Tfs.flatten(),Rs.flatten())
                #         LTEs                            =dmg_inst.get_LTEs_w_penalties(dTs.flatten(),Tfs.flatten(),Rs.flatten())
                #         min_panel_LTEs_alt_opt, min_tube_LTEs_alt_opt   =dmg_inst.calc_minimum_panel_LTEs(model_lite, LTEs)
                #         # LTEs_options[i_y,i_x] = np.average(min_panel_LTEs_alt_opt)  #  mean panel lifetime method
                #         # LTEs_options[i_y,i_x] = np.sqrt(np.mean((min_panel_LTEs_alt_opt - LTE_trigger)**2)) # min RMSE method
                #         # LTE_option              =  LTE_rsme = np.sqrt(np.mean((LTEs - LTE_rmse)**2)) # total RMSE method
                #         LTE_option              = np.min(LTEs) # min method
                #         LTEs_options.append(LTE_option)
                #         coord_list.append([i_y,i_x])
                #         # del(model_inst) # model edits
                # LTEs_options =np.array(LTEs_options)
                        ### 
                        # # again, don't forget to put the x,y grid points in context of the total grid!
                        iterable_dict ={}
                        y_move_opt  =y_ind_top+i_y
                        x_move_opt  =x_ind_left+i_x
                        flux_arr_moved_kW  =helpers_heuristic.move_2d_array_from_ctr(flux_arr_og_kW, x_move_opt, y_move_opt)
                        # update the receiver flux profile with the new flux
                        flux_profile_option_W    =flux_profile + flux_arr_moved_kW*1e3
                        flux_profile_option_kW   =flux_profile_option_W/1e3

                        ## solve lightweight thermal model
                        flux_profile_option_kW          =therm_helpers.increase_flux_resolution_blocked(flux_low_res = flux_profile_option_kW, ndim_new = HALOS_res)
                        thermal_clone                   =copy.deepcopy(model_lite_checkin_clone)    # the model can be intiialized with the current flux profile
                        dmg_clone                       =copy.deepcopy(dmg_inst)

                        iterable_dict['thermal_clone']      =thermal_clone
                        iterable_dict['dmg_clone']          =dmg_clone
                        iterable_dict['flux_profile_option']=flux_profile_option_kW
                        iterable_dict['RMSE_std']           =LTE_rmse

                        iterable.append(iterable_dict)
                        coord_list.append([i_y,i_x])
                        
                # use multiprocessing to accelerate assessment
                cpus                =multiprocessing.cpu_count()
                p                   =multiprocessing.Pool(cpus)
                LTEs_options_grid   =np.array(p.map(flux_to_LTE_w_MP, iterable )) 
                del(p)

            ##

                ## find the best LTE spot, look at options immediately adjacent
                # assuming metric is average of panel min LTEs
                max_index   =np.argmax(LTEs_options_grid)
                max_dy      =coord_list[max_index][0]
                max_dx      =coord_list[max_index][1]

                y_move  =y_ind_top+max_dy  
                x_move  =x_ind_left+max_dx

                x_search_range =[x_move-1, x_move+1]
                y_search_range =[y_move-1, y_move, y_move+1]   # assuming an increment of 2, we have already searched the two options directly above and below the current option
                ## i'm looking at the x's and we already evaluated the dashes if our increment is 2:
                ##  x   -   x
                ##  x start x
                ##  x   -   x
                adj_move_list       =[] # this is an absolute coord list
                LTEs_options_adj    =[]
                for x in x_search_range:
                    for y in y_search_range: 
                        y_move_adj        =y
                        x_move_adj        =x
                        flux_image_adj    =helpers_heuristic.move_2d_array_from_ctr(flux_arr_og_kW, x_move_adj, y_move_adj)
                        # update the receiver flux profile with the new flux
                        flux_profile_option_adj_W   =flux_profile + flux_image_adj*1e3 # flux profile is in W
                        flux_profile_option_adj_kW  =flux_profile_option_adj_W/1e3
                        ## solve lightweight thermal model
                        adj_thermal_clone           =copy.deepcopy(model_lite_checkin_clone)    # the model can be intiialized with the current flux profile
                        flux_profile_option_adj_kW  =therm_helpers.increase_flux_resolution_blocked(flux_low_res = flux_profile_option_adj_kW, ndim_new = HALOS_res)
                        therm_helpers.solve_LWT_thermal_model(adj_thermal_clone,flux_profile_option_adj_kW)
                        dTs_adj, Tfs_adj, qabs_adj, Rs_adj      =therm_helpers.get_thermal_results(adj_thermal_clone)
                        LTEs_adj                                =dmg_inst.get_LTEs_w_penalties(dTs_adj.flatten(),Tfs_adj.flatten(),Rs_adj.flatten())
                        min_panel_LTEs_adj, min_tube_LTEs_adj   =dmg_inst.calc_minimum_panel_LTEs(adj_thermal_clone, LTEs_adj)
                        # LTEs_options[i_y,i_x] = np.average(min_panel_LTEs_alt_opt)  #  mean panel lifetime method
                        # LTEs_options[i_y,i_x] = np.sqrt(np.mean((min_panel_LTEs_alt_opt - LTE_trigger)**2)) # min RMSE method
                        # LTE_option              =  LTE_rsme = np.sqrt(np.mean((LTEs - LTE_rmse)**2)) # total RMSE method
                        min_panel_LTEs_adj          =np.clip(min_panel_LTEs_adj, -1e7, LTE_rmse)  # only give the low lifetimes a voice
                        LTE_option_adj            = np.average(min_panel_LTEs_adj) # average method
                        LTEs_options_adj.append(LTE_option_adj)
                        adj_move_list.append([y_move_adj,x_move_adj])
                        del(adj_thermal_clone) # model edits
                
                LTEs_options_adj     =np.array(LTEs_options_adj)                           # get rid of model clone so that we can restart next time
                max_adj_option_index =np.argmax(LTEs_options_adj)

                # update x_move and y_move if adjacent options are better
                if LTEs_options_grid[max_index] < LTEs_options_adj[max_adj_option_index]:
                    y_move =adj_move_list[max_adj_option_index][0]
                    x_move =adj_move_list[max_adj_option_index][1]          


            ## ------ update the flux profile ---------------------------------------
            # update the receiver flux profile with finalized image placement. If Touts are reached then no reason to do this
            flux_arr_moved_kW   =helpers_heuristic.move_2d_array_from_ctr(flux_arr_og_kW, x_move, y_move)
            flux_profile        =flux_profile + flux_arr_moved_kW*1e3
            placement_info      =[key, int(x_move), int(y_move)]
            placements.append(placement_info)
            ## -------------------------------------------------------------


            ## ------ update tracking info -------------------------------------------
            ## solve and check LTEs every 10 images until we hit a flux violation, then solve after every image placement
            if (i % 10 ==0) or is_flux_violation:
                if i > 0: # no need to delete clone if first iteration
                    del(model_lite_checkin_clone)      # get rid of old model so that thermal model always initializes identically
                model_lite_checkin_clone        =copy.deepcopy(model_lite)  # create a model clone that has the same initialization
                flux_profile_checkin_kW         =flux_profile/1e3           # checkin flux is the current flux profile on the receiver 
                flux_profile_checkin_kW         =therm_helpers.increase_flux_resolution_blocked(flux_low_res = flux_profile_checkin_kW, ndim_new = HALOS_res) # might need to increase flux resolution to match thermal model
                
                therm_helpers.solve_LWT_thermal_model(model_lite_checkin_clone,flux_profile_checkin_kW)
                dTs_checkin, Tfs_checkin, qabs_checkin, Rs_checkin  =therm_helpers.get_thermal_results(model_lite_checkin_clone)
                LTEs_checkin                                        =dmg_inst.get_LTEs_w_penalties(dTs_checkin.flatten(),Tfs_checkin.flatten(),Rs_checkin.flatten())
                min_panel_LTEs_checkin, min_tube_LTEs_checkin       =dmg_inst.calc_minimum_panel_LTEs(model_lite_checkin_clone, LTEs_checkin)


            ## check for lifetime violations in each panel
            is_flux_violation   =np.min(min_panel_LTEs_checkin) < LTE_trigger        
            ## -----------------------------------------------------------------------

            # redo the image placement with lower sigma x if a sub-threshhold LTE occurs
            if endgame:
                in_loop =False
            elif (np.min(min_panel_LTEs_checkin) < LTE_threshold) and (iteration < miter):
                in_loop =True
                fx      =np.max([0,fx-0.1])         # decrement fx but if fx reaches zero then stop decrementing
                if (fx <= 0.0) and (fy > 0):
                    fy  =np.max([0,fy-0.1])         # decrement fy but if fy reaches zero then stop decrementing
                elif (fx <= 0.0) and (fy <= 0.0):   # we've tried everything! Only remaining place to put images is on the edge
                    print("entering edge placement mode")
                    endgame =True

                ## ------ remove current image from the flux profile to avoid negative LTE ---------------------------------------
                flux_profile        =flux_profile - flux_arr_moved_kW*1e3
                placements.pop()
                print('negative LTE encountered, relaxing sigma x requirements')
                ## ----------------------------------------------------------------------------------------------------------------
            elif (np.min(min_panel_LTEs_checkin) < LTE_threshold) and (iteration > miter):
                print('--- exceeded max. number of iterations. Skipping image ---')
                in_loop =False
            else: # while loop exit criteria allows image advancement if none of the panels have negative LTE
                in_loop =False 

        # conditional logic to decide what receiver flowpath to focus the next image on
        if flowpath_config == '2_ctr':
            if not model_lite.tubes[0][0].flow_against_gravity: # assuming symmetry. If the left most panel is flowing w gravity then both outlet temps are at bottom of first/last array column
                T_out_left  =Tfs_checkin[Tfs_checkin.shape[0]-1,0]
                T_out_right =Tfs_checkin[Tfs_checkin.shape[0]-1,-1]
            else: # assuming symmetry. If the left most panel is against gravity then both outlet temps are at top of first/last array column
                T_out_left  =Tfs_checkin[0,0]
                T_out_right =Tfs_checkin[0,-1]

            # if multiple tubes/panel, need to condense into a single temperature
            T_out_left      =np.average(T_out_left)
            T_out_right     =np.average(T_out_right)

            if (T_out_left >= T_out_obj) and (T_out_right < T_out_obj):
                print(f'target outlet temperature reached on left side: {T_out_left} C')
                is_left_done    =True
                is_right_done   =False
            elif (T_out_left < T_out_obj) and (T_out_right >= T_out_obj):
                print(f'target outlet temperature reached on right side: {T_out_right} C')
                is_left_done    =False
                is_right_done   =True
            elif (T_out_left >= T_out_obj) and (T_out_right >= T_out_obj):
                print(f'target outlet temperatures reached on both sides: {T_out_left} and {T_out_right} C')
                is_left_done    =True
                is_right_done   =True
                break
            elif (T_out_left < T_out_obj) and (T_out_right < T_out_obj): # keep switching sides until one of the outlet temperatures is done
                is_left_done    =not is_left_done
                is_right_done   =not is_right_done

    # ------------------------ end of loop -------------------------

    t_loop_end  =timeit.default_timer()
    print(f'total loop time: {t_loop_end-t_loop_start}')


    ## get some in-depth results
    flux_profile_kW_results   =flux_profile/1e3
    flux_profile_kW_results   =therm_helpers.increase_flux_resolution_blocked(flux_low_res = flux_profile_kW_results, ndim_new = HALOS_res*6)
    therm_helpers.solve_LWT_thermal_model(model_hvy,flux_profile_kW_results)
    dTs_results, Tfs_results, qabs_results, Rs_results  =therm_helpers.get_thermal_results(model_hvy)
    LTEs_results                                        =dmg_inst.get_LTEs_w_penalties(dTs_results.flatten(),Tfs_results.flatten(),Rs_results.flatten()) # could also use get_LTEs, which just returns zero if SR reached
    min_panel_LTEs_results, min_tube_LTEs_results       =dmg_inst.calc_minimum_panel_LTEs(model_hvy, LTEs_results)

    # estimate total thermal power delivered
    Qabs_total  =np.average(flux_profile_kW_results)*H*W*model_hvy.solar_abs
    print(f'total absorbed power power estimate from flux profile is: {Qabs_total} kW')

    print('lifetimes in years: \n')
    print(min_panel_LTEs_results)
    print('\n')

    print('all tube lifetimes in years: \n')
    print(min_tube_LTEs_results)
    print('\n')

    print(f'last image key index was:{i}')
    print('\n')

    if flowpath_config == '2_ctr':
        if not model_lite.tubes[0][0].flow_against_gravity: # assuming symmetry. If the left most panel is flowing w gravity then both outlet temps are at bottom of first/last array column
            T_out_left  =Tfs_results[Tfs_results.shape[0]-1,0]
            T_out_right =Tfs_results[Tfs_results.shape[0]-1,-1]
    else: # assuming symmetry. If the left most panel is against gravity then both outlet temps are at top of first/last array column
            T_out_left  =Tfs_results[0,0]
            T_out_right =Tfs_results[0,-1]

    print(f'outlet temperatures:{T_out_left} and {T_out_right} C')

    # plot result
    cbr_fontsize =12
    fig,ax =plt.subplots()
    im =ax.imshow(flux_profile_kW_results, extent =[-model_lite.D/2,model_lite.D/2,0,model_lite.H])
    ax.set_xlabel('x location (m)')
    ax.set_ylabel('y location (m)')
    divider = make_axes_locatable(ax)  
    cax = divider.append_axes("right", size="2%", pad=0.08)     # thanks stack exchange!
    cbr = fig.colorbar(im, cax=cax)
    cbr.set_label(label='incident flux (kW/m$^2$)', size=cbr_fontsize)
    cbr.ax.tick_params(labelsize=cbr_fontsize)
    fig.savefig(f'imgs/{casename}_flux_profile',dpi=300)
    plt.show()
    plt.close()

    # ## make a results report dictionary and save it as a json file 
    if flowpath_config == '2_ctr':
        results_dict                    ={}
        results_dict['Qsinc_W']         =np.average(flux_profile)*H*W
        results_dict['inc_flux_max']    =model_hvy.operating_conditions.inc_flux.max()
        results_dict['trigger']         =LTE_trigger
        results_dict['mflow']           =model_hvy.operating_conditions.mass_flow
        results_dict['T_out_right']     =np.average(T_out_right) # NOTE: this could be made more accurate by including tube edge factors instead of weighting each one equally
        results_dict['T_out_left']      =np.average(T_out_left)
        # calculate the design power assuming constant cp
        mflow_left                      =model_hvy.operating_conditions.mass_flow[0]
        mflow_right                     =model_hvy.operating_conditions.mass_flow[1]
        T_in_C                          =model_hvy.Tfin_design-273.15
        cp_avg                          =tube.fluid.cp(( ((290+565)/2) + 273.15))
        results_dict['Qfluid_W']        =mflow_left*cp_avg*(np.average(T_out_left) - T_in_C) + mflow_right*cp_avg*(np.average(T_out_right) - T_in_C)
        results_dict['N_heliostats_used']=i
        results_dict['min_tube_LTEs']   =min_tube_LTEs_results.tolist()
        results_dict['offset_factor']   =f_og
        results_dict['cut_Hstats']      =ncut
        results_dict['LTEs']            =LTEs_results.tolist()
        results_dict['dTs']             =dTs_results.tolist()
        results_dict['Tfs']             =Tfs_results.tolist()
        
        ## get all inputs from json input file
        inputName = receiver_filestring + '.json'
        outputName= f'reports/{casename}_results.json'

        with open(inputName,) as f:
            report_dict=json.load(f)
        ## combine dictionaries
        report_dict.update(results_dict)
        ## put both dictionaries in a new file
        with open(outputName, "w") as f:
            json.dump(report_dict, f)

        ## save all the placements, too
        with open(f'reports/{casename}_image_placements.json', "w") as f:
            json.dump(placements, f)