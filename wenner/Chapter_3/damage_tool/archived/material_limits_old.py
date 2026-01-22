"""
created by: jwenner on 4/28/2025
class for identifying corrosion velocity and temperature limits for solar receiver materials
temperatures are in C
Tcrit is defined as the maximum allowable film temperature

source for H230, 740H, 800H, A625, 316H: material selection for solar central receiver tubes by Laporte et al. (2021), which compiled other sources

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd
from pathlib import Path

class materialLimits:
    def __init__(self,mat):
        self.mat = mat  # receiver alloy material. Can be: 'A230', 'A617', '316H', '800H', '740H', 'A282', 'A625'
        self.interpolator = None
        if self.mat == 'A230':
            self.Tcrit = 650 #(C) og source: McConohy & Kruizenga (2014)
            # determine where this file is, and add data library path
            py_loc =Path(__file__).resolve()
            self.file_string_SR = py_loc.parent / 'dmg_tool_data/A230_SR_region.csv'
        elif self.mat == '740H': 
            self.Tcrit = 650 #(C) Laporte (2021) decided on 650 because comparable alloy composition to other materials that have proven limit of 650 (C)
        elif self.mat == '800H':
            self.Tcrit = 650 #(C) og source: Bradshaw & Goods (2001)
        elif self.mat == 'A625':
            self.Tcrit = 630 #(C) og source: Miliozzi et al. (2001)
        elif self.mat == '316H':
            self.Tcrit = 600 #(C) og source: Miliozzi et al. (2001)
        elif self.mat == 'A617':
            self.Tcrit = 650 #(C) no source. Judgement call based on common comparison to A230
        elif self.mat =='A282':
            self.Tcrit = 650 #(C) Mark Messner doesn't know of any sources for this yet. He recommended keeping it the same as 740H 
        else:
            print('material not recognized')
        return
        
    def estimate_Tfilm(self,Tcrown,Tf,R):
        """
        estimates the film temperature, where "film" is defined as surface inner crown surface temeprature similar to Laporte (2024)
        Tcrown - (C) crown tempeature
        Tf     - (C) fluid temperature
        R      - resistance ratio (deltaT_cond/deltaT_conv)
        """
        deltaT_total = Tcrown - Tf # total temperature difference
        Tfilm = Tf + deltaT_total/(1+R) #based on definition of R and definition of deltaT_total
        return Tfilm
    
    def check_T_pts(self,T_pts):
        """
        compares the given temperature 
        Tpts - array (C)
        returns True if Tcheck is above the limit
        returns False if Tcheck is below the limit
        """
        toasty_array = np.array( T_pts > self.Tcrit)

        return toasty_array
    
    def getPlotPts(self,R,xLow=100,xHigh=600,res=100): 
        """
        returns plot pts for corrosion limit based on thermal assumptions
        R      - resistance ratio (deltaT_cond/deltaT_conv)
        res    - number of points along the line
        """
        deltaT_totals = np.linspace(xLow,xHigh,res)
        Tfs = self.Tcrit - deltaT_totals/(1+R)
        return deltaT_totals, Tfs
    
    def make_SR_interpolator(self):
        self.load_lookup_table()
        self.generate_RBF_interpolator()

    def load_lookup_table(self):
        self.df_SR = pd.read_csv(self.file_string_SR)

    def generate_RBF_interpolator(self):
        dTs_og = self.df_SR.columns[1:].values.astype(float)
        Tfs_og = self.df_SR['Tf'].values

        dTs = np.tile(dTs_og, Tfs_og.shape)
        Tfs = np.repeat(Tfs_og, dTs_og.shape)
        data_coords = np.dstack( (dTs,Tfs)).squeeze()
        data_vals   = self.df_SR.iloc[:,1:].values.flatten()
        # interp_f = scipy.interpolate.RectBivariateSpline(dTs,Tfs,data_vals.transpose(),kx=1,ky=1)
        interp_f = scipy.interpolate.RBFInterpolator( data_coords, data_vals, neighbors=20, kernel='linear')
        self.interpolator = interp_f

    def check_SR(self, dTs, Tfs):
        """
        Returns True if the ratio of elastic to peak plastic stress is greater than 2
        dT and Tf arrays must be the same size and in C
        """
        if self.interpolator == None:
            self.make_SR_interpolator()
        interp_SRs = self.interpolator(np.dstack( (dTs,Tfs)).squeeze() )
        resets = np.array( interp_SRs > 2)
        return resets
    
    def penalize_SR(self, LTEs, dTs, Tfs):
        """
        further augments LTEs by changing each zero to a penalty value equal to the pythagorean distance between the closest SR point. Bigger penalty for more likely S.R.
        --
        LTEs - a nz*ntubes long flat array of lifetimes in years
        dTs  - a size like LTEs in celsius
        Tfs  - a size like LTEs in celsius
        """
        # get the SR curve coordinates
        dTs_SR = self.df_SR.columns[1:].values.astype(float)
        Tfs_SR = self.df_SR['Tf'].values

        # make a function to calculate the distance for a single dT, Tf combo
        def SR_penalty(dT,Tf):
            all_distances =np.sqrt( ((dTs_SR - dT))**2 + ((Tfs_SR - Tf))**2 ) # calculates the distance between dT, Tf point and all SR coordinates
            min_distance  =np.min(all_distances)
            return -np.exp(min_distance/20)    # we want the minimum distance because this is most accurate, being the closest SR dT, Tf point
        
        # find the indices of where LTEs are set to zero
        indices_zero =np.where(LTEs == 0)[0]
        for index in indices_zero:
            LTEs[index]=SR_penalty(dTs[index], Tfs[index])

        return LTEs

if __name__ == "__main__":
    mat = materialLimits('A230')
    print('film temp is',mat.estimate_Tfilm(Tcrown=520,Tf=400,R=0.5) )
    xPts, yPts = mat.getPlotPts(R=0.5)
    
    fig,ax=plt.subplots()
    ax.plot(xPts,yPts)
    plt.show()