# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:55:41 2024

@author: david.c.doughty.civ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rs_tools as rt
rt.plot_format(10)
from copy import deepcopy
from scipy.ndimage import median_filter
import os
import itertools
import io
import re
from scipy.signal import medfilt
import lmfit as lm
import fit_spc as fs
import warnings
from scipy.sparse import SparseEfficiencyWarning

## Suppress spsolve warning. 
# The als code uses sparse matrix math. The current implementation is not
# the most efficient implementation, python currently throws a warning and converts
# to the default format, which may be slow. For now we suppress this warning
# 
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

#file2load = 'dg_20221117_1304.txt'
#df = pd.read_csv(file2load,comment='#',sep='\t',parse_dates=[1])
outputdir = ''

def plot_fit_ax(myax,x_values,y_values,df_init,mydf,xlims=[800,2000],ymax_mult=1.1,mytext='',color_range=['C2','C3','C4','C5','C6'],plot_raw=False,myyticks=[],stop=False):

    meas_spc_sub=y_values-x_values*mydf.bkr_s.values-mydf.bkr_c.values

    
    max_meas = np.nanmax(meas_spc_sub[((x_values >= xlims[0]) & (x_values <= xlims[1]))])
    
    xv  = x_values[rt.find_nearest(x_values,xlims[0]):rt.find_nearest(x_values,xlims[1])]
    
    #xv = np.arange(xlims[0],xlims[1])
    yv = np.zeros(xv.shape)
    myax.plot(x_values,meas_spc_sub,color='C0',label='obs')  
    for index,mypeak in df_init.iterrows():
        pk_nm = mypeak.pk_name
        if ((mypeak.pk_type == 'Lorentzian') | (mypeak.pk_type == 'Gaussian')):
            pk_c = mydf[str(pk_nm+'_center')].values
            pk_s = mydf[str(pk_nm+'_sigma')].values
            pk_a = mydf[str(pk_nm+'_amplitude')].values
            pk_m = fs.lorentzian(xv,pk_c,pk_s,pk_a)
        elif ((mypeak.pk_type == 'BreitWignerFano')):
            pk_c = mydf[str(pk_nm+'_center')].values
            pk_s = mydf[str(pk_nm+'_sigma')].values
            pk_a = mydf[str(pk_nm+'_amplitude')].values
            pk_f = mydf[str(pk_nm+'_amplitude')].values
            pk_m = fs.breitwignerfano(xv,pk_c,pk_s,pk_a,pk_f)            
            
            
        myax.plot(xv,pk_m,color=color_range[index],label=pk_nm,linestyle='--')
        yv = yv + pk_m
    
    maxy = np.nanmax(np.append(yv,max_meas))
    
    myax.plot(xv,yv,color='C1',linewidth=1,label='fit')        
    myax.legend(handlelength=1,loc='upper right')
    myax.set_xlim(xlims)
    myax.text(0.05,0.85,mytext,transform=myax.transAxes)
    myax.set_ylim([0,maxy*ymax_mult])
    myax.set_yticks(myyticks)
    
    mystr = mytext + f"\tResidual:{mydf.resi_mean.values[0]}\tMSE:{mydf.mse.values[0]}\tRedChi2:{mydf.chisq_red.values[0]}"
    print(mystr)

    if stop == True:
        import pdb
        pdb.set_trace()    



def clean_spc_ids(my_spc_ids):
    #Very specific code to handle text in the input file name we want to remove. 
    #import pdb
    for j in range(len(my_spc_ids)):
        #pdb.set_trace()
        selected_spc_id = my_spc_ids[j]
        my_spc_ids[j] = selected_spc_id.replace('als_e6','').replace('als_e5','').replace('als_e4','')
        #print(my_spc_ids[j],my_spc_ids[j].replace('als_e6','').replace('als_e5','').replace('als_e4',''))
    return my_spc_ids
        

def change_yax_pos(ax):
    vlen,hlen = ax.shape
    rightax = hlen-1
    for i in range(vlen):
        ax[i,rightax].yaxis.set_label_position("right")
        ax[i,rightax].yaxis.tick_right()
 
        
def fine_format_axis(myax,diag_len=0.02):
    #Removes vertical axes bars
    axis_rows,axis_columns = myax.shape
    diag_x = [1-diag_len,1+diag_len]

    for myrow in range(axis_rows):
        for mycol in range(axis_columns):
            try:
                myax_lp = myax[myrow,mycol]
                myax_lp.spines['top'].set_visible(False)
                myax_lp.spines['right'].set_visible(False)
                myax_lp.spines['left'].set_visible(False)
                
                

                if mycol < axis_columns-1:
                    #Add diagonal slash to start of break
                    #myax_lp.plot(diag_x,[-diag_len,diag_len],transform=myax_lp.transAxes,color='k',clip_on=False)
                    myax_lp.plot([1,1],[-diag_len,diag_len],transform=myax_lp.transAxes,color='k',clip_on=False)
                    
                if mycol > 0:
                    #Remove left spine if it's not the first column
                    #myax_lp.spines['left'].set_visible(False)
                    #Add left axis slash if it's not the first column
                    #myax_lp.plot([-diag_len,diag_len],[-diag_len,diag_len],transform=myax_lp.transAxes,color='k',clip_on=False)
                    myax_lp.plot([0,0],[-diag_len,diag_len],transform=myax_lp.transAxes,color='k',clip_on=False)

                else:

                    #myax_lp.plot([0,0],[-diag_len,0.90],transform=myax_lp.transAxes,color='k',clip_on=False)
                    myax_lp.plot([0,0],[0,1],transform=myax_lp.transAxes,color='k',clip_on=False)

            except Exception as err:
                import pdb
                pdb.set_trace()
                
xlims=[800,2000]
xlims_plot = [1000,2000]
q_load_arr = [
        '-lume6',
        '-lume5',
        '-lume4',
        '-lumfit',
        '-nolum',
        '-lsh1',
        '-lg',
        '']

q_load_extra = [
        'als_e6',
        'als_e5',
        'als_e4',
        '',
        '',
        '',
        '',
        '']
    
    
q_comp=7
q_include = [0,1,2,3,5,6,7]
n_qplt = len(q_include)

q_plot_legend = [
        'als e6',
        'als e5',
        'als e4',
        'lumfit',
        'linsh1(1001-1748)',
        'linlg(799-1898)',
        'lin(901-1797)',
        'meas',
        'meas-lin']

marker = ['<','>','^','s','*','p','X','o']
plot_colors = ['cyan','C1','C2','C3','C4','C5','C6','C8']

plot_colors_qinclude = [plot_colors[myqind] for myqind in q_include]

q_text_base = 'dg_2pk_lor-apprx'

q_init_val_data = []

filename = "varlum_results.csv"
with open(filename,'w') as fid:
    fid.write(f"spc_id1,spc_id2,init_q1,init_q2,init_gam_1,init_gam_2,")

#These are the spectra to actually show in the final plot. 
display_spectra = []

'''
Loop over each function - For each function plot all resultant fcns (possibly just with g), color changing
Also plot result q vs Input q, colored by chi squared. (Possibly input functions - but we may need to write code to display them. 
'''

######
#
#   Get the list of figures
#


all_spc_ids = np.array(['selected_151_20160825_212530_6.0_19.0.txt', #old 11, New 1
                        'selected_96919_20160825_001238_0.0_31.0.txt', #old 12, new 2
                        'selected_30881_20160825_120528_44.0_37.0.txt', #old 9, new 3
                        'selected_31624_20160825_184739_26.0_32.0.txt', # old 10, new 4
                        'selected_1070_20160825_212530_48.0_14.0.txt', #old 1, new 5
                        'selected_88054_20160825_234758_18.0_10.0.txt', #old2, new6  
                        'selected_10952_20160825_205359_24.0_40.0.txt', #old3, new7
                        'selected_44428_20160825_095858_32.0_32.0.txt',#old14, new S2
                        'selected_43052_20160825_180008_22.0_20.0.txt',#old 15, new S3
                        'selected_50840_20160825_152129_8.0_20.0.txt',#old13, new S4
                        'selected_68027_20160825_160911_2.0_3.0.txt', #old8, new S5
                        'selected_8837_20160825_094326_34.0_37.0.txt', #old 4, new S6 Good
                        'selected_16909_20160825_222850_38.0_13.0.txt',#old 5, nes S7
                        'selected_4424_20160825_191905_44.0_24.0.txt', #old 6, new S8
                        'selected_10485_20160825_205359_4.0_13.0.txt',]) #old 7, new S9

import matplotlib.cm
cmap = matplotlib.cm.get_cmap('jet')
q_max = 5
input_dir = 'data_files'
fit_result_dir = 'fit_results'

def add_jitter(y_values,sigma=0.1,method='add'):
    #This will give a little unormal jitter to some values
    #adds normal jitter. 
    #We truncate at 'rth*sigma'
    #Currently method = 'add' -> goal here 
    mylen = len(y_values)
    rth=2
    rand_vals = sigma*np.random.randn(mylen)
    rand_vals[rand_vals > rth*sigma] =     rth*sigma
    rand_vals[rand_vals < -1*rth*sigma] = -1*rth*sigma
    
    jitter_vals = np.array([y_values[k] + rand_vals[k] for k in range(mylen)])
    return jitter_vals

d1_c = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
d1_s = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
d1_a = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
g_c = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
g_s = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
g_a = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
all_h = np.full(len(all_spc_ids),np.nan)
all_h1 = np.full(len(all_spc_ids),np.nan)
all_slp = np.full(len(all_spc_ids),np.nan)
all_cst = np.full(len(all_spc_ids),np.nan)

#Uncertainties
g_c_unc = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
d_c_unc = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
h1_unc = np.full(len(all_spc_ids),np.nan)


spc_id_list_1 = np.array(['selected_44428_20160825_095858_32.0_32.0.txt',
                          #'selected_151_20160825_212530_6.0_19.0.txt',
                          'selected_43052_20160825_180008_22.0_20.0.txt',
                          'selected_96919_20160825_001238_0.0_31.0.txt',
                          'selected_31624_20160825_184739_26.0_32.0.txt']
                           )

fig7 = plt.figure(figsize=(8.5,11))
gs = fig7.add_gridspec(4,3,wspace=0.05,hspace=0.1)

fig8,ax8 = plt.subplots(nrows=4,ncols=2,figsize=(8.5,11))



rowylim = [0,1]
rowylim_offsets = 0.1
rowpointlims = [rowylim[0] + rowylim_offsets, rowylim[1]-rowylim_offsets]
rowylim[1] = rowylim[1] + 4*rowylim_offsets


#Indices for the ALS fit
ind_s = 133
ind_e = 647

ind_splt = 213
ind_eplt = 577

fig7_cntr = 0

'''
plt_labs = [['A.','B.','C.','D.'],
           ['E.','F.','G.','H.'],
           ['I.','J.','K.','L.'],
           ['M.','N.','O.','P.'],
           ['Q.','R.','S.','T.']]
'''
plt_labs = [['A.','F.','K.','L.'],
           ['B.','G.','M.','N.'],
           ['C.','H.','O.','P.'],
           ['D.','I.','Q.','R.'],
           ['E.','J.','S.','T.']]

err_sigma=3 #How many standard deviations

labs_Fig8 = ['A.','B.','C.','D.','E.']


text_unc_params = ['g_center' , 'd1_center']

for i in range(len(all_spc_ids)):

    f,ax = plt.subplots(nrows=1,ncols=4,figsize=[18,6])
    
    myinputfile = os.path.join(input_dir,all_spc_ids[i])
    myinputfile_base = all_spc_ids[i]
    
    #print("-------------------------------------------------")
    #print(myinputfile)
    mydata = np.loadtxt(myinputfile,delimiter=' ',skiprows=0)
    data_x = mydata[:,0]
    data_y = mydata[:,1]
    
    isInPlot = True if all_spc_ids[i] in spc_id_list_1 else False
    
    if isInPlot == True:
        mylabels = plt_labs[fig7_cntr]
        gplt_0= fig7.add_subplot(gs[fig7_cntr,0])
        gplt_1= fig7.add_subplot(gs[fig7_cntr,1])
        
        hplt_0 = ax8[fig7_cntr,0]
        hplt_1 = ax8[fig7_cntr,1]

        gs00 = gs[fig7_cntr,2].subgridspec(2,1)
        gplt_2a = fig7.add_subplot(gs00[0])
        gplt_2b = fig7.add_subplot(gs00[1])
        fig7_cntr = fig7_cntr + 1
        gplt_0.plot(data_x,data_y,color='k')
        hplt_0.plot(data_x,data_y,color='k')

    data_y_flt = medfilt(data_y,21)
    
    bkr_e6 = np.zeros(data_y.shape)
    bkr_e5 = np.zeros(data_y.shape)
    bkr_e4 = np.zeros(data_y.shape)
    bkr_e6[ind_s:ind_e] = rt.background_als_core_nu(data_y_flt[ind_s:ind_e],handle_end=True,p=0.001,lmb=1e6)
    bkr_e5[ind_s:ind_e] = rt.background_als_core_nu(data_y_flt[ind_s:ind_e],handle_end=True,p=0.001,lmb=1e5)        
    bkr_e4[ind_s:ind_e] = rt.background_als_core_nu(data_y_flt[ind_s:ind_e],handle_end=True,p=0.001,lmb=1e4)     
    bkr_max = np.nanmax(bkr_e4)

    ax[3].plot(data_x,bkr_e6,color=plot_colors[0])
    ax[3].plot(data_x,bkr_e5,color=plot_colors[1])
    ax[3].plot(data_x,bkr_e4,color=plot_colors[2])
    
    #loop through all q values values
    for j in range(len(q_load_arr)):
        
        ##########################
        #Here load the raw fit file
        #print("       ---------------")
        #print(f"       {q_load_arr[j]}")
        #import pdb
        #pdb.set_trace()
        raw_spcid_json = all_spc_ids[i].replace('.txt','').replace('.0','0') + q_load_extra[j] + '.json'

        rawdir = f"{q_text_base}{q_load_arr[j]}"
        rawfile = f"modelresult_{q_text_base}{q_load_arr[j]}_{raw_spcid_json}"
        rawpath = os.path.join(rawdir,rawfile)
        my_fit_result = lm.model.load_modelresult(rawpath) 
        
        
        g_c_unc[j,i] = my_fit_result.params['g_center'].stderr*err_sigma
        d_c_unc[j,i]  = my_fit_result.params['d1_center'].stderr*err_sigma
        best_fit = my_fit_result.best_fit
        
  
        
        result_unc = my_fit_result.eval_uncertainty(sigma=err_sigma)
        #Checked that my_fit_result.eval equals yv -> does appear to, except limits are a bit different. 
        ###### 
        
        myfitfile = q_text_base + q_load_arr[j] + '.csv'
        mydf_init,mydf_fit = fs.read_fit_result(os.path.join(fit_result_dir,myfitfile))
        
        full_spc_ids = clean_spc_ids(mydf_fit.spc_id.copy())
        #please_stop_here()
        
        mydf_fit_spc = mydf_fit[full_spc_ids == all_spc_ids[i]]

        xv  = data_x[rt.find_nearest(data_x,xlims[0]):rt.find_nearest(data_x,xlims[1])]
        yv = np.zeros(xv.shape) 

        for index,mypeak in mydf_init.iterrows():
            pk_nm = mypeak.pk_name
            if (mypeak.pk_type == 'Lorentzian'):
                pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values
                pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values
                pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values
                pk_m = fs.lorentzian(xv,pk_c,pk_s,pk_a)
            else:
                raise ValueError("you are trying to load a model that is not here....")

            len_s = len(pk_s)
            len_c = len(pk_c)
            len_a = len(pk_a)
            
            slen = ((len_s == 1) & (len_c == 1) & (len_a ==1))
            #Check that the lengths are 1 - > then we can index from 0
            
            if ((pk_nm == 'd1') & (slen == True)):
                d1_c[j,i] = pk_c[0]
                d1_s[j,i] = pk_s[0]
                d1_a[j,i] = pk_a[0]
                if j == 0:
                    d1_base_c = pk_c[0]
                    d1_base_s = pk_s[0]
                    d1_base_a = pk_a[0]
            elif slen == False:
                raise ValueError("Lengths not matching")
                
                
            if ((pk_nm == 'g') & (slen == True)):
                g_c[j,i] = pk_c[0]
                g_s[j,i] = pk_s[0]
                g_a[j,i] = pk_a[0]
                if j == 0:
                    g_base_c = pk_c[0]
                    g_base_s = pk_s[0]
                    g_base_a = pk_a [0]             
            elif slen == False:
                raise ValueError("Lengths not matching")                
                        
            yv = yv + pk_m    

        myalpha=0.9
        mycolor=plot_colors[j]
        
        mylinestyle = '-'
        if j == 4:
            mylinestyle == '--'
        ax[0].plot(xv,yv,alpha=myalpha,color=mycolor,linewidth=2,linestyle=mylinestyle)


        if j == 4:
            ax[3].plot([],[],color=plot_colors[j],linestyle=mylinestyle)

            
        elif j>2:
            y_bkr = data_x*mydf_fit_spc.bkr_s.values+mydf_fit_spc.bkr_c.values
            ax[3].plot(data_x,y_bkr,color=plot_colors[j])
            
            if isInPlot == True: 
                gplt_0.plot(data_x,y_bkr,color=plot_colors[j])
                hplt_0.plot(data_x,y_bkr,color=plot_colors[j])
            
            if j == 3:
                bkr_min = np.nanmin(y_bkr[ind_splt:ind_eplt])
            
        if j == 7:
            all_h[i] = np.nanmax(data_y[300:550])
            all_h1[i] = np.nanmax(yv)
            h1_unc_retrieve = result_unc[best_fit == np.max(best_fit)]
            if len(h1_unc_retrieve) == 1:
                h1_unc[i]  = h1_unc_retrieve[0] #If the length is 1
            else:
                raise IndexError("Tried to index h1_unc but nothing found here")
            all_slp[i] = mydf_fit_spc.bkr_s.values[0]
            all_cst[i] = mydf_fit_spc.bkr_c.values[0]
            
        if ((isInPlot == True) & (j != 4)): 
            gplt_1.plot(xv,yv,alpha=myalpha,color=mycolor,linewidth=2,linestyle=mylinestyle)
            hplt_1.plot(xv,yv,alpha=myalpha,color=mycolor,linewidth=2,linestyle=mylinestyle)
        if isInPlot == True :
            myloadarr = q_load_arr[j].strip("-")
            print(f"{mylabels[0]},{all_spc_ids[i]},{myloadarr},{d1_c[j,i]:.1f},{g_c[j,i]:.1f}")
            
    with open(filename,'a') as fid:
        fid.write(f"{all_spc_ids[i]},spc_id2,init_q1,init_q2,init_gam_1,init_gam_2,")
    minlim1=0.995
    minlim2=0.8
    maxlim1 = 1.005
    maxlim2 = 1.2
    d1a_min = np.nanmin(np.delete(d1_a[:,i],4))*minlim2
    d1a_max = np.nanmax(np.delete(d1_a[:,i],4))*maxlim2
    d1s_min = np.nanmin(np.delete(d1_s[:,i],4))*minlim1
    d1s_max = np.nanmax(np.delete(d1_s[:,i],4))*maxlim1   
    d1c_min = np.nanmin(np.delete(d1_c[:,i],4))*minlim1
    d1c_max = np.nanmax(np.delete(d1_c[:,i],4))*maxlim1   
    ga_min = np.nanmin(np.delete(g_a[:,i],4))*minlim2
    ga_max = np.nanmax(np.delete(g_a[:,i],4))*maxlim2
    gs_min = np.nanmin(np.delete(g_s[:,i],4))*minlim1
    gs_max = np.nanmax(np.delete(g_s[:,i],4))*maxlim1
    gc_min = np.nanmin(np.delete(g_c[:,i],4))*minlim1
    gc_max = np.nanmax(np.delete(g_c[:,i],4))*maxlim1

    ax[1].scatter(d1_a[:,i],d1_c[:,i],color=plot_colors)
    
    #ax[1].set_xlim([d1a_min,d1a_max])
    ax[1].set_ylim([d1c_min,d1c_max])
    ax[2].scatter(g_a[:,i],g_c[:,i],color=plot_colors)    
    #ax[2].set_xlim([ga_min,ga_max])
    ax[2].set_ylim([gc_min,gc_max])    
    ax[3].plot(data_x,data_y,color='k')
    ax[3].set_xlim(xlims)
    ax[3].legend(q_plot_legend)
    ax[3].set_ylim([bkr_min*0.8,bkr_max*1.5])
    meas_spc_sub=data_y-data_x*mydf_fit_spc.bkr_s.values-mydf_fit_spc.bkr_c.values
    ax[0].plot(data_x,meas_spc_sub,'k',linewidth=2,alpha=0.5)
    
    ax[0].legend(q_plot_legend)
    maxv = np.nanmax(meas_spc_sub[ind_splt:ind_eplt]*1.2)
    ax[0].set_ylim([0,maxv])
    ax[0].set_xlim(xlims)
    ax[0].set_title('Fit Function')
    ax[0].set_xlabel('Wavenumber \\ cm-1')
    
    if isInPlot == True:

        mc='C0'
        gplt_1.plot(data_x,meas_spc_sub,'k',linewidth=2,zorder=-999)
        gplt_0.plot(data_x,bkr_e6,color=plot_colors[0])
        gplt_0.plot(data_x,bkr_e5,color=plot_colors[1])
        gplt_0.plot(data_x,bkr_e4,color=plot_colors[2])
        hplt_1.plot([],[],'k',linewidth=2)
        hplt_1.plot(data_x,meas_spc_sub,color=[0.5,0.5,0.5],linewidth=2,zorder=-999)
        hplt_0.plot(data_x,bkr_e6,color=plot_colors[0])
        hplt_0.plot(data_x,bkr_e5,color=plot_colors[1])
        hplt_0.plot(data_x,bkr_e4,color=plot_colors[2])

        
        d1c_diff = np.subtract(d1_c,d1_c[q_comp,:])
        gc_diff = np.subtract(g_c,g_c[q_comp,:])
        
        d1c_diff[d1c_diff > 20] = 20
        d1c_diff[d1c_diff < -20] = -20
        gc_diff[gc_diff > 20] = 20
        gc_diff[gc_diff < -20] = -20
        #import pdb
        #pdb.set_trace()
        scatter_points = np.linspace(rowpointlims[0],rowpointlims[1],len(d1c_diff[q_include,i]))
        gplt_2a.scatter(d1c_diff[q_include,i],scatter_points,color=plot_colors_qinclude,alpha=0.8,edgecolor='k')
        gplt_2b.scatter(gc_diff[q_include,i],scatter_points,color=plot_colors_qinclude,edgecolor='k')
        
        gplt_1.set_ylim([0,maxv])
        gplt_0.set_xlim(xlims_plot)
        gplt_1.set_xlim(xlims_plot)
        gplt_1.set_yticks([])
        
        hplt_1.set_ylim([0,maxv])
        hplt_0.set_xlim(xlims_plot)
        hplt_1.set_xlim(xlims_plot)
        hplt_1.set_yticks([])
        
        
        myxticks = [1100,1350,1600,1850]
        gplt_0.set_xticks(myxticks)
        gplt_1.set_xticks(myxticks)
        hplt_0.set_xticks(myxticks)
        hplt_1.set_xticks(myxticks)
        
        gplt_0.text(0.05,0.9,mylabels[0],transform = gplt_0.transAxes)
        gplt_1.text(0.05,0.9,mylabels[1],transform = gplt_1.transAxes)
        
        
        hplt_0.text(0.05,0.9,mylabels[0],transform = hplt_0.transAxes)
        
        gplt_2a.text(0.05,0.8,mylabels[2],transform = gplt_2a.transAxes)
        gplt_2b.text(0.05,0.8,mylabels[3],transform = gplt_2b.transAxes)
        gplt_2a.text(0.95,0.8, "D Peak Center",horizontalalignment='right',transform = gplt_2a.transAxes)
        gplt_2b.text(0.95,0.8,"G Peak Center",horizontalalignment='right',transform = gplt_2b.transAxes)
        
        gplt_0.set_ylim([0,1.1*np.nanmax(data_y[ind_s:ind_e])])
        hplt_0.set_ylim([0,1.1*np.nanmax(data_y[ind_s:ind_e])])

        gplt_2a.set_xlim([-22,22])
        gplt_2b.set_xlim([-22,22])
        gplt_2a.set_ylim(rowylim)
        gplt_2b.set_ylim(rowylim)
        gplt_2a.set_xticks([-21,-10,0,10,21])
        gplt_2b.set_xticks([-21,-10,0,10,21])
        gplt_2a.set_xticklabels([])
        gplt_2b.set_xticklabels([])
   
        
        gplt_2a.tick_params(top=False,bottom=True,right=False,left=False)#,labelbottom=False,labeltop=True,right=True,left=False,labelleft=False,labelright=True)
        gplt_2b.tick_params(top=False,bottom=True,right=False,left=False)
        #gplt_2a.set_xticks([1300,1340,1380])


        gplt_0.set_ylabel("Intensity\n \\Arb. Units")
        hplt_0.set_ylabel("Intensity\n \\Arb. Units")
        hplt_1.legend(q_plot_legend,bbox_to_anchor=(0.955555,0.98),ncol=1,loc='upper center',fontsize=8)


        if fig7_cntr == 1:
            gplt_0.set_title("Luminescence\nApproximations")
            gplt_1.set_title("Fitted Function")
            gplt_2a.set_title("Fitted D/G position")
            gplt_1.legend(q_plot_legend,bbox_to_anchor=(0.5, 1.4),ncol=4,loc='center')
            hplt_0.set_title("Luminescence\nApproximations")
            hplt_1.set_title("Fitted Function")
            #hplt_1.legend(q_plot_legend,bbox_to_anchor=(0.5, 1.4),ncol=4,loc='center')
        #if fig7_cntr  != 1:
            #gplt_2a.set_xticklabels('')

        if fig7_cntr < len(spc_id_list_1):
            #gplt_2b.set_xticklabels('')
            gplt_0.set_xticklabels('')
            gplt_1.set_xticklabels('')
            hplt_0.set_xticklabels('')
            hplt_1.set_xticklabels('')
        if fig7_cntr == len(spc_id_list_1):
            gplt_1.set_ylim([0,6000])
            #gplt_1.set_xticklabels(['1000','1250','1500','1750'])
            gplt_1.set_xlabel(r'Wavenumber \ cm$^{-1}$') 
            gplt_0.set_xlabel(r'Wavenumber \ cm$^{-1}$') 
            gplt_2b.set_xlabel(r'$\Delta$ Wavenumber (from lin) \cm$^{-1}$')
            xtl = ['<-20','-10','0','10','>20']
            hplt_1.set_ylim([0,6000])
            #gplt_1.set_xticklabels(['1000','1250','1500','1750'])
            hplt_1.set_xlabel(r'Wavenumber \ cm$^{-1}$') 
            hplt_0.set_xlabel(r'Wavenumber \ cm$^{-1}$') 
            #gplt_2a.set_xticklabels(xtl)
            gplt_2b.set_xticklabels(xtl)
            
           
        gplt_2a.set_yticks([])
        gplt_2b.set_yticks([])


    
    #max_meas = np.nanmax(meas_spc_sub[((x_values >= xlims[0]) & (x_values <= xlims[1]))])        

    
    outtext = 'Extra_lor_varlum_' + all_spc_ids[i].replace('selected','').replace('.txt','.png')
    plt.savefig(outtext)
    #please_stop_here()
    #plt.close('all')
    plt.close(f)  
    
fig8.subplots_adjust(hspace=0.07,wspace=0.02)
fig7.tight_layout()
fine_format_axis(ax8)
#fig8.tight_layout()
fig7.savefig("FigS16_varlumsummary.png")
fig8.savefig("Fig8_varlumsummary.png")

#please_stop_here()

d1_h = fs.amplitude_to_height(d1_a,d1_s)
g_h = fs.amplitude_to_height(g_a,g_s)


q_load_arr = [
        '-lume6',
        '-lume5',
        '-lume4',
        '-lumfit',
        '-nolum',
        '-lsh1',
        '-lg',
        '']


q_comp=7
q_include = [0,1,2,3,5,6,7]
n_qplt = len(q_include)

q_plot_legend = [
        'als e6',
        'als e5',
        'als e4',
        'lumfit',
        'linsh1',
        'linlg',
        'lin',
        'meas/lin']

marker = ['<','>','^','s','*','p','X','o']
plot_colors = ['cyan','C1','C2','C3','C4','C5','C6','C8']

plot_colors_qinclude = [plot_colors[myqind] for myqind in q_include]


marker = ['<','>','^','s','*','p','X','o']
plot_colors = ['cyan','C1','C2','C3','C4','C5','C6','C8']
q_text_base = 'dg_2pk_lor-apprx'


number_backgrounds = len(q_load_arr)
nrm = number_backgrounds-1

ctr = 1500
cst_h = all_slp*ctr + all_cst



dqtxst = ''.join([f"d{myq}," for myq in q_load_arr])
gqtxst = ''.join([f"g{myq}," for myq in q_load_arr])

maxhrat = np.divide(all_h1,cst_h)

maxhrat_unc = np.divide(h1_unc,cst_h) #Assumes zero error on the background
plt.close('all')
f3,ax3 = plt.subplots(nrows=1,ncols=2,figsize=(14,8))
with open(filename,'w') as f:
    
    f.write("i,spc_id,maxhrat," + dqtxst+gqtxst + "\n")
    
    for i in range(15):
        g_str = ''.join([f"{myg}," for myg in g_c[:,i]])
        d_str = ''.join([f"{myd}," for myd in d1_c[:,i]])
        f.write(f"{i},{all_spc_ids[i]},{maxhrat[i]}," + d_str + g_str+ "\n")    


for myind,mycolor in zip(q_include,plot_colors_qinclude):
    print(myind,mycolor)
    
    d1rh = np.divide(d1_h[myind,:],cst_h)
    grh = np.divide(g_h[myind,:],cst_h)

    ax3[0].errorbar(d1_c[myind,:],maxhrat,
                    yerr=maxhrat_unc,xerr=g_c_unc[myind,:],
                    marker=marker[myind],
                    linestyle='none',
                    color=plot_colors[myind],
                    markeredgecolor='k',
                    elinewidth=1,
                    capsize=3)
    
    ax3[1].errorbar(g_c[myind,:],maxhrat,
                    yerr=maxhrat_unc,xerr=g_c_unc[myind,:],
                    marker=marker[myind],
                    linestyle='none',
                    color=plot_colors[myind],
                    markeredgecolor='k',
                    elinewidth=1,
                    capsize=3)
 
ax3[0].legend(q_plot_legend[0:-1])  
ax3[0].text(0.05,0.95,"A.",transform = ax3[0].transAxes)
ax3[1].text(0.25,0.95,"B.",transform = ax3[1].transAxes)
ax3[0].set_title(r'Max intensity/1500 cm$^{-1}$ luminescence vs D1 Center')   
ax3[0].set_ylabel(r'Max intensity/1500 cm$^{-1}$ luminescence')
ax3[0].set_xlabel(r' D1 Center \ cm$^{-1}$')
ax3[1].legend(q_plot_legend[0:-1]) 
ax3[1].set_title(r'Max intensity/1500 cm$^{-1}$ luminescence vs G Center') 
ax3[1].set_ylabel(r'Max intensity/1500 cm$^{-1}$ luminescence')
ax3[1].set_xlabel(r'G Center \ cm$^{-1}$') 
f3.subplots_adjust(hspace=0,wspace=0.1)
f3.savefig("Fig9_RelheightvsCenter.png")

