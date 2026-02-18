# -*- coding: utf-8 -*-
"""

Key point here - we use this code to generate the output files!!


"""

#filename

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rs_tools as rt
rt.plot_format(10)
from copy import deepcopy
from scipy.ndimage import median_filter
import fit_spc as ft

def process_rs(rs):
    #This would be used for doing any preprocessing to the rs
    #Data saved as PNG appear to be 0-1. We convert back to ADC values (65535).
    adc_conv = 65535
    rs = rs *adc_conv
    return rs

def convert_rs_weight(rs):
    #USed if you want to calculate error on a RS. This is used to weight the fit model if 'weights' is set to 'true'. 
    #Here, we take in the "Raw" raman spectrum, and compute the error as sqrt(I/3) where I is the raman intensity. 
    # Note that we return the weight:1/w
    nreps=3
    wght_cmp= np.sqrt(np.true_divide(rs,nreps))
    wght = np.true_divide(1,wght_cmp)
    return wght


mult = 65535

xmin = 213
xmax=511


###################################################
#
#   Optionally - load data, plot data and fits to data. 
#
#   Key point: we do not want to assume that the numbers line up. 
load_data = False
if load_data == True:
    file_rdat = 'alldata_r13_TimeSeries_2_ClCsBkr.rdat'
    file_rmta = 'alldata_r13_TimeSeries_2_ClCsBkr.rmta'
    rdatdata = np.loadtxt(file_rdat,delimiter=',')
    dt_meta = pd.read_table(file_rmta,sep=',')
    rdatdata_nobk = np.loadtxt('alldata_r13_TimeSeries_0_Cl.rdat',delimiter=',')
    
x_values = rdatdata[0,:]
raman_values = rdatdata[1::,:]*mult

raman_values_nobk = rdatdata_nobk[1::,:]*mult

wn_fit = x_values[xmin:xmax]


####################################################
#
#
#   Selected Examples
#
#
#
#
#



allinds_d = [(63471,''),
           (44428,''),
           (12281,''),
           (36982,''),
           (96065,''),
           (31624,''),
           (16909,''),
           (1070,''),
           (88054,''),
           (50840,''),
           (10485,''),
           (68027,''),
           (151,''),
           (4424,''),
           (8837,''),
           (10952,''),
           (36980,''),
           (21248,''),
           (3914,''),
           (20031,''),
           (10420,''),
           (57832,''),
           (4494,''),
           (68027,''),
           (43052,''), 
           (30881,''), 
           (96919,'')]
 


#Generate output file for fitting code.    
for i,myind in enumerate(allinds_d):
    
    myRaman = raman_values[myind[0]]
    myRaman_raw = raman_values_nobk[myind[0]]
    myweights = convert_rs_weight(myRaman_raw)
    myx_values = x_values
    myDG = dt_meta.iloc[myind[0]]
    all_data = np.hstack((myx_values[:,np.newaxis],myRaman[:,np.newaxis],myweights[:,np.newaxis]))
    my_serialdate = str(myDG.DateTime).replace(' ','_').replace(":","").replace("-","")
    
    f,ax = plt.subplots()
    ax.plot(myx_values,myRaman)
    axa = ax.twinx()
    axa.plot(myx_values,myRaman_raw,'C1')
    ax.set_xlim([800,1900])
    f.savefig(f'selected_v1_{myind[0]}.png')
    
    fname = f'selected_{myind[0]}_{my_serialdate}_{myDG.x:.1f}_{myDG.y:.1f}.txt'
    np.savetxt(fname,all_data,delimiter=' ')
