"""
damage tool created by jwenner on 7/10/2025 

goal is to make SRLIFE-generated FEA results accessible and useful for predicting solar receiver tube creep-fatigue damage and lifetime
"""
import numpy as np
import math
import pandas as pd
import scipy.interpolate 
import timeit
import sys
sys.path.append('./../heuristic_dev')
import material_limits
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

class damageTool():
    def __init__(self, mat_string):
        if mat_string == 'A230':
            self.mat        ='A230'
            # determine where this file is, and add data library path
            py_loc =Path(__file__).resolve()
            self.file_string = py_loc.parent / 'dmg_tool_data/A230_LTE_tables.csv'

            print('NOTE! as of 7/28/2025 this damage table provides LTE in units of years')
        else:
            print('unknown material')
        self.materialLimits = material_limits.materialLimits('A230')
        self.interpolator =None
        self.df_LTE       =pd.DataFrame()
        self.ctf          =None

        
    def plot_dmg_map():
        pass

    def make_interpolator(self):
        """
        oversight function
        """
        self.load_lookup_table()
        self.generate_RBF_interpolator()

    def load_lookup_table(self):
        self.df_LTE = pd.read_csv(self.file_string)

    # def grid_interp_LTEs(self,xi):
    #     """
    #     NOTE: untested
    #     """
    #     finite_mask = np.isfinite(self.df_LTE['LTE'].values) # infinite lifetime values do not effect interpolation
    #     # make a subselection for lookup table
    #     dTs = self.df_LTE['dT'].values[finite_mask]
    #     Tfs = self.df_LTE['Tf'].values[finite_mask]
    #     LTEs= self.df_LTE['LTE'].values[finite_mask]

    #     interp_grid = scipy.interpolate.griddata(np.dstack( (dTs,Tfs) ).squeeze(), LTEs, xi, method='linear')

    def generate_RBF_interpolator(self):
        """
        creates a radial basis function interpolator based on finite lifetime data
        -tested on 7/28/25 and returned accurate values. Note that transitiion to infinite LTE as seen on maps is not captured because of nature of RBF...
            ... this means that the data may predict infnite lifetime but this interpolator will still provide a finite (albeit extremely high) number
        """
        if self.df_LTE.empty == True:
            self.load_lookup_table()

        finite_mask = np.isfinite(self.df_LTE['LTE'].values) # infinite lifetime values do not effect interpolation
        # make a subselection for lookup table
        dTs = self.df_LTE['dT'].values[finite_mask]
        Tfs = self.df_LTE['Tf'].values[finite_mask]
        LTEs= self.df_LTE['LTE'].values[finite_mask]

        interp_f = scipy.interpolate.RBFInterpolator(np.dstack(  (dTs,Tfs)).squeeze(), LTEs, neighbors=100, kernel='linear')
        
        self.interpolator = interp_f # returns time in years

    def make_contour_function_from_interpolator(self,LTE_desired=30, cutoff=270, show_LTE_ctr=False):
        """
        creates contour levels using the interpolator - this ensures that the desired and predicted LTE agree
        
        """
        if self.interpolator == None:
            self.make_interpolator()
            print('making interpolator')
        # create and plot a damage map using LTEs from the interpolator
        map_fig,map_ax   =self.make_dmg_map_from_interpolator(LTE_desired)
        # select a desired contour, get vertices, show on plot

        # get the desired LTE index from the level list
        LTE_index =self.level_list.index(LTE_desired)
        # grab the contour's vertices
        all_dT_verts    =self.ctf.get_paths()[LTE_index].vertices[:,0]
        all_Tf_verts    =self.ctf.get_paths()[LTE_index].vertices[:,1]
        # make subselection, assuming you want the left most contour
        Tf_data_max =self.df_LTE['Tf'].max()
        slice_indices     =np.where(all_Tf_verts == Tf_data_max)[0]
        if slice_indices.size > 1:
            dTs_LTE             =all_dT_verts[0:slice_indices[1]]
            Tfs_LTE             =all_Tf_verts[0:slice_indices[1]]
        else: 
            dTs_LTE             =all_dT_verts[0:slice_indices[0]]
            Tfs_LTE             =all_Tf_verts[0:slice_indices[0]]
        dTs_LTE             =np.where(dTs_LTE < cutoff, dTs_LTE,cutoff) # prevent dTs from venturing into stress reset zone
        
        # plot the contour function
        map_ax.plot(dTs_LTE, Tfs_LTE)

        dT_function =scipy.interpolate.interp1d(Tfs_LTE, dTs_LTE, fill_value='extrapolate')
        self.dT_function = dT_function
        print('dT function created')
        if show_LTE_ctr:
            plt.show()
        plt.close()

        return 
    
    @staticmethod
    def fmt(x):
        # for formatting the damage map
        s = f"{x:.0f}" + ' (yrs)'
        return s

    def make_dmg_map_from_interpolator(self, user_level):
        """
        makes a damage map for contour access and visualization. Note: based on RBF interpolator, so slightly different values than previously done
        ---
        user_level - desired lifetime contour in years
        """
        print('making damage map and returning contours')
        # initialize plot
        fig, ax=plt.subplots()

        # make a meshgrid of x,y values for interpolation use
        Tf_data_min =self.df_LTE['Tf'].min()
        Tf_data_max =self.df_LTE['Tf'].max()
        dT_data_min =self.df_LTE['dT'].min()
        dT_data_max =self.df_LTE['dT'].max()
        Tf_range  =np.linspace(Tf_data_min,Tf_data_max,250)
        dT_range  =np.linspace(dT_data_min, dT_data_max, 250)
        dT_samples, Tf_samples  =np.meshgrid(dT_range, Tf_range)
        # get the LTE values based on the RBF interpolator
        LTE_samples =self.get_LTEs_no_limits(dT_samples.flatten(), Tf_samples.flatten(), 0.5*np.ones(Tf_samples.size))
        # make the contour plot
        levels=[10.1, 30.1, 100.1]   #[5,10,15,20,25,30,40,50,60,70,80]
        levels.append(user_level)
        levels.sort()
        ctf=ax.contour(dT_range, Tf_range, LTE_samples.reshape(dT_samples.shape), levels=levels,cmap=cm.copper_r,zorder=1) #was using cm.cool or colors='black', levels=levels
        self.ctf=ctf
        self.level_list=levels
        ax.axes.clabel(ctf,inline=True,inline_spacing=1, fontsize=6, fmt=self.fmt,manual=False,levels=levels) #left click: place. center click: done
        fontsize=14
        ax.set_ylabel('fluid temperature (C)', fontsize=fontsize)
        ax.set_xlabel('total temperature difference (C)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)


        return fig,ax
    
    def get_LTEs_no_limits(self,dTs,Tfs,Rs):
        """
        predicts LTEs for each provided dT,Tf coordinate
        dT - array of total temperature differences
        Tf - array of fluid temperatures
        ----
        returns LTE in cycles or years, depending on the material's database. LTE only. Ignores corrosion and SR region
        """
        if self.interpolator == None:
            self.make_interpolator()
            print('making interpolator')
        ## make the x,y points:
        interp_coords = np.dstack(  (dTs,Tfs)).squeeze()
        ## get the values: 
        interp_LTEs = self.interpolator(interp_coords)

        return interp_LTEs

    def get_LTEs(self,dTs,Tfs,Rs):
        """
        predicts LTEs for each provided dT,Tf coordinate
        dT - array of total temperature differences
        Tf - array of fluid temperatures
        ----
        returns LTE in cycles or years, depending on the material's database. Returns a 0 for a coordinate if that coordinate is in the corrosion or stress reset region
        """
        if self.interpolator == None:
            self.make_interpolator()
            print('making interpolator')
        ## make the x,y points:
        interp_coords = np.dstack(  (dTs,Tfs)).squeeze()
        ## get the values: 
        interp_LTEs = self.interpolator(interp_coords)

        ## change any values that violate the film temperature limits
        # get all the crown temperatures
        Tcrowns = Tfs + dTs
        # get all the film temperatures
        Tfilms = self.materialLimits.estimate_Tfilm(Tcrowns,Tfs,Rs)
        # find the points that have T > Tcrit:
        toasty_array = self.materialLimits.check_T_pts(Tfilms)
        # assign the T > Tcrit locations lifetimes of 0 years
        interp_LTEs[toasty_array] = 0

        ## change any values that are in the stress reset region
        # get an array of booleans
        resets = self.materialLimits.check_SR(dTs, Tfs)
        # set the coordinates that are resetting to have a LTE of 0
        interp_LTEs[resets] = 0

        return interp_LTEs
    
    def get_LTEs_w_SR_penalty(self, dTs, Tfs, Rs):
        """
        predicts LTEs for each provided dT,Tf coordinate and oturageously penalizes any stress reset cases
        dT - array of total temperature differences
        Tf - array of fluid temperatures
        ----
        returns LTE in cycles or years, depending on the material's database. Returns a 0 for a coordinate if that coordinate is in the corrosion region, and a negative value for those in the SR region
        """
        if self.interpolator == None:
            self.make_interpolator()
            print('making interpolator')
        ## make the x,y points:
        interp_coords = np.dstack(  (dTs,Tfs)).squeeze()
        ## get the values: 
        interp_LTEs = self.interpolator(interp_coords)


        ## change any values that are in the stress reset region. Penalize them!
        # get an array of booleans
        resets = self.materialLimits.check_SR(dTs, Tfs)
        # set the coordinates that are resetting to have a LTE of 0
        interp_LTEs[resets] = 0
        # for the coordinates that have LTE of zero, assign a penalty
        interp_LTEs         = self.materialLimits.penalize_SR(interp_LTEs, dTs, Tfs)
        
        ## change any values that violate the film temperature limits
        # get all the crown temperatures
        Tcrowns = Tfs + dTs
        # get all the film temperatures
        Tfilms = self.materialLimits.estimate_Tfilm(Tcrowns,Tfs,Rs)
        # find the points that have T > Tcrit:
        toasty_array = self.materialLimits.check_T_pts(Tfilms)
        # assign the T > Tcrit locations lifetimes of 0 years
        interp_LTEs[toasty_array] = 0

        return interp_LTEs
    
    def make_contour_function(self, LTE_desired):
        """
        creates the desired LTE contour for the given material 
        ---
        LTE_desired - (int) the number of years desired. options: 30,40,50,60,70,80
        ---
        returns: the maximum allowable dT total for a given fluid temperature. Input and output are in Celsius 
        """
        file_string = f'dmg_tool_data/{self.mat}_{LTE_desired}yr_ctrs_tempBase.csv'
        verts_df = pd.read_csv(file_string)
        dT_ctr = verts_df['dT'].values
        Tf_ctr = verts_df["Tf"].values
        dT_function =scipy.interpolate.interp1d(Tf_ctr, dT_ctr, fill_value='extrapolate')
        self.dT_function = dT_function
        
        return 

    def calc_minimum_panel_LTEs(self, receiver, LTEs):
        """
        outputs the LTEs per panel, based on the discretization of the receiver model
        ---
        receiver    - NREL thermal model object
        LTes        - single, flattened array with length of nz*ntubes_sim
        ---
        returns:
        min_panel_LTEs  - array with length of nPanels
        tube_min_LTEs   - array with length of ntubes_sim*Npanels
        """
        ## find out the number of tubes and axial discretization
        ntubes_sim  =receiver.ntubesim
        axial_nodes =receiver.tubes[0][0].disc.nz
        Npanels     =receiver.Npanels

        ## reshape, find the minimum for each tube
        LTEs_reshaped=LTEs.reshape(axial_nodes, Npanels, ntubes_sim)
        tube_min_LTEs=LTEs_reshaped.min(axis=0)

        ## find the minimum of each panel
        pmods           =np.mod(np.arange(Npanels*ntubes_sim),ntubes_sim)
        inds            =list(np.where(pmods == 0)[0])
        min_panel_LTEs  =np.array([])
        for i in range(Npanels):
            if i < Npanels-1:                
                min_panel_LTE   =np.min(tube_min_LTEs.flatten()[inds[i]:inds[i+1]])
            else:
                min_panel_LTE   =np.min(tube_min_LTEs.flatten()[inds[i]:])

            min_panel_LTEs  =np.concatenate((min_panel_LTEs,np.array([min_panel_LTE]) ))

        return min_panel_LTEs, tube_min_LTEs
    
    def calc_minimum_panel_LTEs_simple_inputs(self, ntubes_sim, axial_nodes, Npanels, LTEs):
        """
        outputs the LTEs per panel, based on the discretization of the receiver model. Same functionality as above, but doesn't require entire receiver object
        ---
        ntubes_sim  - how many tubes the model simulated per panel
        axial_nodes - (int)
        Npanels     - total number of panels on the receiver
        LTEs        - single, flattened array with length of nz*ntubes_sim
        ---
        returns:
        min_panel_LTEs  - array with length of nPanels
        tube_min_LTEs   - array with length of ntubes_sim*Npanels
        """

        ## reshape, find the minimum for each tube
        LTEs_reshaped=LTEs.reshape(axial_nodes, Npanels, ntubes_sim)
        tube_min_LTEs=LTEs_reshaped.min(axis=0)

        ## find the minimum of each panel
        pmods           =np.mod(np.arange(Npanels*ntubes_sim),ntubes_sim)
        inds            =list(np.where(pmods == 0)[0])
        min_panel_LTEs  =np.array([])
        for i in range(Npanels):
            if i < Npanels-1:                
                min_panel_LTE   =np.min(tube_min_LTEs.flatten()[inds[i]:inds[i+1]])
            else:
                min_panel_LTE   =np.min(tube_min_LTEs.flatten()[inds[i]:])

            min_panel_LTEs  =np.concatenate((min_panel_LTEs,np.array([min_panel_LTE]) ))

        return min_panel_LTEs, tube_min_LTEs

if __name__ == "__main__":
    dmg_obj = damageTool(mat_string='A230')
    dmg_obj.load_lookup_table()
    xi = np.array([[102,300]])

    dmg_obj.generate_RBF_interpolator()
    t_interp_start = timeit.default_timer()
    LTE_xi = dmg_obj.interpolator(xi)
    t_interp_end = timeit.default_timer()
    print(f'time to interpolate was: {t_interp_end-t_interp_start}')
    print(f'interpolated LTE was:{LTE_xi} years')

    ### test ability to look up lifetimes
    t_interp_start = timeit.default_timer()
    dmg_obj2 = damageTool(mat_string='A230')
    LTEs_returned = dmg_obj2.get_LTEs(np.array([160,200]),np.array([550,500]),np.array([0.5,0.5]) )
    print('the returned LTEs are:')
    print(LTEs_returned)
    t_interp_end = timeit.default_timer()
    print(f'time to interpolate was: {t_interp_end-t_interp_start}')

    ### test ability to look up lifetimes, but rerun now that interpolator is made
    t_interp_start = timeit.default_timer()
    LTEs_returned = dmg_obj2.get_LTEs(np.array([170,170,170,180]),np.array([500,525,549,549]),np.array([0.5,0.5,0.5,0.5]) )
    print('the returned LTEs are:')
    print(LTEs_returned)
    t_interp_end = timeit.default_timer()
    print(f'time to interpolate was: {t_interp_end-t_interp_start}')

    ### test ability to look up lifetimes for 20 tubes
    t_interp_start = timeit.default_timer()
    npts = 100
    LTEs_returned = dmg_obj2.get_LTEs(np.linspace(100,220,npts),np.linspace(300,550,npts),np.ones(npts)*0.5 )
    print('the returned LTEs are:')
    print(LTEs_returned)
    t_interp_end = timeit.default_timer()
    print(f'time to interpolate was: {t_interp_end-t_interp_start}')
   
    ### test ability to look up lifetimes for 100 tubes but using new SR penalty function
    t_interp_start = timeit.default_timer()
    npts =1000
    xs   =np.linspace(100,340,npts)
    LTEs_returned = dmg_obj2.get_LTEs_w_SR_penalty(xs,400*np.ones(npts),np.ones(npts)*0.5 )
    print('the returned LTEs are:')
    print(LTEs_returned)
    t_interp_end = timeit.default_timer()
    print(f'time to interpolate was: {t_interp_end-t_interp_start}')

    fig, ax=plt.subplots(tight_layout=True)
    ax.plot(xs, LTEs_returned)
    ax.set_xlabel('dT (C)')
    ax.set_ylabel('LTE (years)')
    # fig.savefig('imgs/new_SR_penalty_325')
    plt.show()
    plt.close()

    # make a 30 year contour just for demo
    dmg_obj.make_contour_function(30)
    Tfs     =np.linspace(290,565)
    dTs     =dmg_obj.dT_function(Tfs)

    fig,ax =plt.subplots()
    ax.plot(dTs,Tfs)
    ax.set_xlabel('temperature difference (C)',fontsize=14)
    ax.set_ylabel('fluid temperature (C)',fontsize=14)
    fig.savefig('imgs/30_yr_contour',dpi=300)
    plt.show()

    # make a damage map using RBF interpolator
    dmg_obj.make_contour_function_from_interpolator()


