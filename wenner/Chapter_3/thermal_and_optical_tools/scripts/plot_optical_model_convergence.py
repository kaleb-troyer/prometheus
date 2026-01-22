"""
created by jwenner on 10/19/25 to illustrate effect of ndim resolution on optical flux profiles from copylot

"""

import pandas as pd
import matplotlib.pyplot as plt

df =pd.read_csv('aiming/optical_model_resolution_impact.csv')

W_rec   =18
H_rec   =15
Area    =W_rec*H_rec

npts =df['ndim'].values **2     # calculate total number of points on discretized grid
npts_per_area   =npts/Area      # pts/m2

fontsize =14
fig,ax =plt.subplots()
ax.scatter(npts_per_area, df['q_flux_max'])
ax.set_xlabel('points / m2', fontsize =fontsize)
ax.set_ylabel('peak flux (W/m2)', fontsize =fontsize)
plt.savefig('imgs/peak_inc_flux_convergence_plot',dpi=300)
plt.show()
plt.close()

fig,ax =plt.subplots()
ax.scatter(npts_per_area, df['Q_rec_inc']/1e3)
ax.set_xlabel('points / m2', fontsize=fontsize)
ax.set_ylabel('incident power (MW)', fontsize=fontsize)
plt.savefig('imgs/incident_power_convergence_plot',dpi=300)
plt.show()
plt.close()