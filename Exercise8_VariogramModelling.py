# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 21:07:22 2021

@author: nhtl
"""

import geostatspy.GSLIB as GSLIB                        # GSLIB utilies, visualization and wrapper
import geostatspy.geostats as geostats                  # GSLIB methods convert to Python    

%matplotlib inline
import os                                               # to set current working directory 
import sys                                              # supress output to screen for interactive variogram modeling
import io
import numpy as np                                      # arrays and matrix math
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # plotting
from matplotlib.pyplot import cm                        # color maps
from ipywidgets import interactive                      # widgets and interactivity
from ipywidgets import widgets                            
from ipywidgets import Layout
from ipywidgets import Label
from ipywidgets import VBox, HBox
from matplotlib.patches import Ellipse                  # plot an ellipse

os.chdir("D:\Geostatistics")                                   # set the working directory
data = pd.read_csv("Exercise8_VariogramModelling.csv")         # read a .csv file in as a DataFrame
data['Iron value (%Fe)'] = data['Iron value (%Fe)']/100
data = data.rename(columns={"Iron value (%Fe)":"Iron value (fraction)"})
#print(df.iloc[0:5,:])                                  # display first 4 samples in the table as a preview
data.head()                                             # we could also use this command for a table preview 

data.describe().transpose()                             # summary table of sand only DataFrame statistics


#Let's transform the porosity and permeaiblity data to standard normal (mean = 0.0, standard deviation = 1.0, Gaussian shape). This is required for sequential Gaussian simulation (common target for our variogram models) and the Gaussian transform assists with outliers and provides more interpretable variograms.
#Let's look at the inputs for the GeostatsPy nscore program. Note the output include an ndarray with the transformed values (in the same order as the input data in Dataframe 'df' and column 'vcol'), and the transformation table in original values and also in normal score values.
geostats.nscore                                         # see the input parameters required by the nscore function

data['Normal Iron Value'], tvIron, tnsIron = geostats.nscore(data, 'Iron value (fraction)') # nscore transform for all iron values (%Fe) 
data.head()                                                                            # preview DataFrame with nscore transforms

plt.subplot(221)                                        # plot original iron value histograms
plt.hist(data['Iron value (fraction)'], facecolor='red',bins=np.linspace(0.25,0.5,50),histtype="stepfilled",alpha=0.6,density=True,cumulative=True,edgecolor='black',label='Original')
plt.xlim([0.25,0.5]); plt.ylim([0,1.0])
plt.xlabel('Iron value (fraction)'); plt.ylabel('Frequency'); plt.title('Iron Value')
plt.legend(loc='upper left')
plt.grid(True)

plt.subplot(222)  
plt.hist(data['Normal Iron Value'], facecolor='green',bins=np.linspace(-3,3,50),histtype="stepfilled",alpha=0.4,density=True,cumulative=True,edgecolor='black',label = 'Transform')
plt.xlim([-2.5,2.5]); plt.ylim([0,1.0])
plt.xlabel('Iron value (fraction)'); plt.ylabel('Frequency'); plt.title('Nscore Iron Value')
plt.legend(loc='upper left')
plt.grid(True)

plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=2.2, wspace=0.2, hspace=0.3)
plt.show()

cmap = plt.cm.plasma                                   # color map
plt.subplot(121)
GSLIB.locmap_st(data,'Easting (feet)','Northing (feet)','Iron value (fraction)',0,1000,0,700,0,0.5,'Iron value','Easting (feet)','Northing (feet)','Iron value',cmap)

plt.subplot(122)                                       # location map of normal score transform
GSLIB.locmap_st(data,'Easting (feet)','Northing (feet)','Normal Iron Value',0,1000,0,700,-3,3,'Gaussian Transformed Iron Value','Easting (feet)','Northing (feet)','Gaussian Transformed Iron Value',cmap)
plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.0, wspace=0.5, hspace=0.3)
plt.show()

vmap, npmap = geostats.varmapv(data,'Easting (feet)','Northing (feet)','Normal Iron Value',tmin=-9999,tmax=9999,nxlag=10,nylag=10,dxlag=50,dylag=50,minnp=1,isill=1)

plt.subplot(121)
GSLIB.pixelplt_st(vmap,-500,500,-350,350,1.0,0,1.2,'Nscore Iron Value Variogram Map','Easting (feet)','Northing (feet)','Nscore Variogram',cmap)

plt.subplot(122)
GSLIB.pixelplt_st(npmap,-500,500,-350,350,1.0,0,50,'Nscore Iron Value Variogram Map Number of Pairs','Easting (feet)','Northing (feet)','Number of Pairs',cmap)

plt.subplots_adjust(left=0.0, bottom=0.3, right=2.0, top=1.0, wspace=0.2, hspace=3)
plt.show()

print('The shape of the output is ' + str(vmap.shape))

tmin = -9999.; tmax = 9999.                             # no trimming 
lag_dist = 100.0; lag_tol = 100.0; nlag = 10;            # maximum lag is 700m and tolerance > 1/2 lag distance for smoothing
bandh = 9999.9; atol = 22.5                             # no bandwidth, directional variograms
isill = 1                                               # standardize sill
azi_mat = [0,22.5,45,67.5,90,112.5,135,157.5]           # directions in azimuth to consider

# Arrays to store the results
lag = np.zeros((len(azi_mat),nlag+2)); gamma = np.zeros((len(azi_mat),nlag+2)); npp = np.zeros((len(azi_mat),nlag+2));

for iazi in range(0,len(azi_mat)):                      # Loop over all directions
    lag[iazi,:], gamma[iazi,:], npp[iazi,:] = geostats.gamv(data,'Easting (feet)','Northing (feet)','Normal Iron Value',tmin,tmax,lag_dist,lag_tol,nlag,azi_mat[iazi],atol,bandh,isill)
    plt.subplot(4,2,iazi+1)
    plt.plot(lag[iazi,:],gamma[iazi,:],'x',color = 'black',label = 'Azimuth ' +str(azi_mat[iazi]))
    plt.plot([0,2000],[1.0,1.0],color = 'black')
    plt.xlabel(r'Lag Distance $\bf(h)$, (feet)')
    plt.ylabel(r'$\gamma \bf(h)$')
    plt.title('Directional NSCORE Iron Value Variogram')
    plt.xlim([0,900])
    plt.ylim([0,4])
    plt.legend(loc='upper left')
    plt.grid(True)

plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=4.2, wspace=0.2, hspace=0.3)
plt.show()

# Select the plot above for the major and minor
imajor = 4
iminor = 0

print('Major direction is ' + str(azi_mat[imajor]) + ' azimuth.')
print('Minor direction is ' + str(azi_mat[iminor]) + ' azimuth.')

if not abs(azi_mat[imajor] - azi_mat[iminor]) == 90.0:
    print('Major and minor directions must be orthogonal to each other.')
    sys.exit()

plt.subplot(1,2,1)
plt.plot(lag[imajor,:],gamma[imajor,:],'x',color = 'black',label = 'Azimuth ' + str(azi_mat[imajor]))
plt.plot([0,2000],[1.0,1.0],color = 'black')
plt.xlabel(r'Lag Distance $\bf(h)$, (feet)')
plt.ylabel(r'$\gamma \bf(h)$')
plt.title('Directional NSCORE Iron Value Variogram - Major ' + str(azi_mat[imajor]) + ' Azimuth')
plt.xlim([0,900])
plt.ylim([0,3])
plt.legend(loc='upper left')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(lag[iminor,:],gamma[iminor,:],'x',color = 'black',label = 'Azimuth ' +str(azi_mat[iminor]))
plt.plot([0,2000],[1.0,1.0],color = 'black')
plt.xlabel(r'Lag Distance $\bf(h)$, (feet)')
plt.ylabel(r'$\gamma \bf(h)$')
plt.title('Directional NSCORE Iron Value Variogram  - Minor ' + str(azi_mat[iminor]) + ' Azimuth')
plt.xlim([0,900])
plt.ylim([0,3])
plt.legend(loc='upper left')
plt.grid(True)

plt.subplots_adjust(left=0.0, bottom=0.0, right=2.2, top=1.6, wspace=0.2, hspace=0.3)
plt.show()

nug = 0.0; nst = 2                                             # 2 nest structure variogram model parameters
it1 = 3; cc1 = 0.8; azi1 = azi_mat[imajor]; hmaj1 = 750; hmin1 = 250
it2 = 3; cc2 = 0.2; azi2 = azi_mat[imajor]; hmaj2 = 700; hmin2 = 400

vario = GSLIB.make_variogram(nug,nst,it1,cc1,azi1,hmaj1,hmin1,it2,cc2,azi2,hmaj2,hmin2) # make model object
nlag = 80; xlag = 15;                                          # project the model in the 045 azimuth
index_maj,h_maj,gam_maj,cov_maj,ro_maj = geostats.vmodel(nlag,xlag,azi_mat[imajor],vario)                                                     # project the model in the 135 azimuth
index_min,h_min,gam_min,cov_min,ro_min = geostats.vmodel(nlag,xlag,azi_mat[iminor],vario)

plt.subplot(1,2,1)
plt.plot(lag[imajor,:],gamma[imajor,:],'x',color = 'black',label = 'Azimuth ' +str(azi_mat[imajor]))
plt.plot([0,2000],[1.0,1.0],color = 'black')
plt.plot(h_maj,gam_maj,color = 'black')
plt.xlabel(r'Lag Distance $\bf(h)$, (feet)')
plt.ylabel(r'$\gamma \bf(h)$')
plt.title('Directional NSCORE Iron Value Variogram - Major ' + str(azi_mat[imajor]) + ' Azimuth')
plt.xlim([0,900])
plt.ylim([0,3])
plt.legend(loc='upper left')

plt.subplot(1,2,2)
plt.plot(lag[iminor,:],gamma[iminor,:],'x',color = 'black',label = 'Azimuth ' +str(azi_mat[iminor]))
plt.plot([0,2000],[1.0,1.0],color = 'black')
plt.plot(h_min,gam_min,color = 'black')
plt.xlabel(r'Lag Distance $\bf(h)$, (feet)')
plt.ylabel(r'$\gamma \bf(h)$')
plt.title('Directional NSCORE Iron Value Variogram  - Minor ' + str(azi_mat[iminor]) + ' Azimuth')
plt.xlim([0,900])
plt.ylim([0,3])
plt.legend(loc='upper left')

plt.subplots_adjust(left=0.0, bottom=0.0, right=2.2, top=1.6, wspace=0.2, hspace=0.3)
plt.show()

