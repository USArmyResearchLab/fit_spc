# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:43:50 2024

@author: david.c.doughty.civ
"""

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
from scipy.signal import medfilt,savgol_filter

#file2load = 'dg_20221117_1304.txt'
#df = pd.read_csv(file2load,comment='#',sep='\t',parse_dates=[1])
outputdir = 'figures-focus'

def gaussian(xv,x0,sigma,A):
    #This is the function as defined by lmfit

    gaus = (A/(sigma*np.sqrt(2*np.pi)))*np.exp(-1*np.divide((xv-x0)**2,2*sigma**2))
    #Something not right about the gausssian. 
    return gaus

def lorentzian(xv,x0,sigma,A):
    lor = (A/np.pi)*np.divide(sigma,(xv-x0)**2+sigma**2)
    return lor

def breitwignerfano(xv,x0,sigma,A,q):
    bwf = (A*((q*sigma/2)+xv-x0)**2)/((sigma/2)**2+(xv-x0)**2)#(A/np.pi)*np.divide(sigma,(xv-x0)**2+sigma**2)
    
    return bwf

def read_fit_result(myfile,skipinirows=3):
    #Reader for the output files, reads into two data frames. 
    #Helped by: https://stackoverflow.com/questions/39724298/pandas-extract-comment-lines
    with open(myfile,'r') as f:
        header_itr = itertools.takewhile(lambda myline:myline.startswith('#'),f)
        header = list(header_itr)
    initialization_list = header[skipinirows::]
    initialization_list_clean = [myline[1::] for myline in initialization_list]
    
    mydf_init = pd.read_csv(io.StringIO('\n'.join(initialization_list_clean)),delim_whitespace=True)
    
    mydf_fit = pd.read_csv(myfile,comment='#',sep=',',skipfooter=1,engine='python')
    
    #From: https://stackoverflow.com/questions/42171709/creating-pandas-dataframe-from-a-list-of-strings   
    return mydf_init,mydf_fit


def amplitude_to_height(amplitude,sigma,q=0,gamma=0,curve='Lorentzian'):
    if curve == 'Lorentzian':
        height= amplitude*0.3183/sigma
    elif curve == 'Gaussian':
        height= amplitude*0.3989/sigma
    else:
        print("height calc not added yet....defaulting to gaussian")
        height=np.nan
    return height

def plot_fit_ax(myax,x_values,y_values,df_init,mydf,xlims=[800,2000],ymax_mult=1.1,mytext='',color_range=['C2','C3','C4','C9','C5'],plot_raw=False,myyticks=[],stop=False):

    meas_spc_sub=y_values-x_values*mydf.bkr_s.values-mydf.bkr_c.values

    
    max_meas = np.nanmax(meas_spc_sub[((x_values >= xlims[0]) & (x_values <= xlims[1]))])

    
    xv  = x_values[rt.find_nearest(x_values,xlims[0]):rt.find_nearest(x_values,xlims[1])]
    
    yv = np.zeros(xv.shape)
    myax.plot(x_values,meas_spc_sub,color='C0',label='obs')  
    for index,mypeak in df_init.iterrows():
        pk_nm = mypeak.pk_name
        if ((mypeak.pk_type == 'Lorentzian') | (mypeak.pk_type == 'Gaussian')):
            pk_c = mydf[str(pk_nm+'_center')].values
            pk_s = mydf[str(pk_nm+'_sigma')].values
            pk_a = mydf[str(pk_nm+'_amplitude')].values
            pk_m = lorentzian(xv,pk_c,pk_s,pk_a)
        elif ((mypeak.pk_type == 'BreitWignerFano')):
            pk_c = mydf[str(pk_nm+'_center')].values
            pk_s = mydf[str(pk_nm+'_sigma')].values
            pk_a = mydf[str(pk_nm+'_amplitude')].values
            pk_f = mydf[str(pk_nm+'_amplitude')].values
            pk_m = breitwignerfano(xv,pk_c,pk_s,pk_a,pk_f)            
            
            
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
        
def add_jitter(y_values):
    jitter_vals = np.zeros(y_values.shape)
    jitter_vals = np.random.rand()
    #Need to add jitter

def change_yax_pos(ax):
    vlen,hlen = ax.shape
    rightax = hlen-1
    for i in range(vlen):
        ax[i,rightax].yaxis.set_label_position("right")
        ax[i,rightax].yaxis.tick_right()
        
xlims=[900,1800]
q_load_arr = [
        '',
        '-noc',
        '-var3',
        '-var2']

q_plot_legend = ['COM','COM-NC','VAR1','VAR2']

init_markers = ['o','<','>','^']
plot_colors = ['C0','C1','C7','C8']
q_text_base = 'dg_5pk_c_smp-ratapprx'

input_dir = 'fit_results'

q_init_val_data = []

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

#filename = 'dg_2pk_lor-apprx.csv'
#df_init_2bl,df_fit_2bl = read_fit_result(filename)
#all_spc_ids = df_fit_2bl.spc_id.values
#Make whichever one you want to be 'all_spc_ids'. 

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


spc_id_list_fig5 = np.array(['selected_8837_20160825_094326_34.0_37.0.txt',
                          'selected_1070_20160825_212530_48.0_14.0.txt',
                          'selected_10952_20160825_205359_24.0_40.0.txt',
                          'selected_43052_20160825_180008_22.0_20.0.txt']
                           )

import matplotlib.cm
cmap = matplotlib.cm.get_cmap('jet')
q_max = 5



d1_c = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
d1_s = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
d1_a = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
g_c = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
g_s = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
g_a = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
g_h = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
d2_c = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
d2_s = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
d2_a = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
d2_h = np.full((len(q_load_arr),len(all_spc_ids)),np.nan)
all_slp = np.full(len(all_spc_ids),np.nan)
all_cst = np.full(len(all_spc_ids),np.nan)


spc_id_list_1 = np.array(['selected_44428_20160825_095858_32.0_32.0.txt',
                          'selected_151_20160825_212530_6.0_19.0.txt',
                          'selected_43052_20160825_180008_22.0_20.0.txt',
                          'selected_96919_20160825_001238_0.0_31.0.txt',
                          'selected_50840_20160825_152129_8.0_20.0.txt']
                           )



#

#ind_start = 133
#ind_end = 647

#213:577

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



fig5 = plt.figure(figsize=[7.5,10])
#left=0.12, bottom=0.08, right=0.85, top=0.92,
gs5 = fig5.add_gridspec(4,3,wspace=0.05,hspace=0.05,left=0.1, bottom=0.25, right=0.9, top=0.99)
fig5_cntr = 0
spc_id_list_fig5 = np.array(['selected_8837_20160825_094326_34.0_37.0.txt',
                          'selected_1070_20160825_212530_48.0_14.0.txt',
                          'selected_10952_20160825_205359_24.0_40.0.txt',
                          'selected_43052_20160825_180008_22.0_20.0.txt']
                           )
lowest_f5_ind = 3
f5A_xlab = [1000,1330,1600]
f5E_xlab = [1000,1200,1400,1600]
f5labels0 = ['A.','B.','C.','D.']
f5labels1 = ['E.','F.','G.','H.']
j_name = ['Com','ComNC','LoGHiD2','SDZ633']
all_linestyles = ['-','-.','--',':']
peak_colors = ['C2','C3','C4','C9','C5']

q_plot_legend = ['Smoothed Raw','G Fit','D1 Fit','D2 Fit','D3 Fit','D4 Fit','Com','ComNC','LoGHiD2','SDZ633','Com','ComNC','HLoGHiD2','SDZ633']
plot_n_rows = 4 #Indexes from zero 
plot_n_columns = 2
fig7 = plt.figure(figsize=[7.5,10])
gs = fig7.add_gridspec(5,3,wspace=0.05,hspace=0.1)
ncntr=0
ccntr=0
tncntr=0
myplotnum=1
myalpha=0.9


plot_names = ["FigS11.png","FigS12.png","FigS13.png"]

for i in range(len(all_spc_ids)):

    myinputfile = all_spc_ids[i]
    load_path = os.path.join("data_files",myinputfile)
    
    add_to_fig5 = myinputfile in spc_id_list_fig5
    if add_to_fig5 == True:
        loc_f5= np.where(myinputfile == spc_id_list_fig5)[0][0]
    
    
    print(myinputfile)
    mydata = np.loadtxt(load_path,delimiter=' ',skiprows=0)
    data_x = mydata[:,0]
    data_y = mydata[:,1]
    

    mylabels = plt_labs[ccntr]
    gplt_0= fig7.add_subplot(gs[ccntr,0])
    gplt_1= fig7.add_subplot(gs[ccntr,1])
    gplt_1_alt = gplt_1.twinx()
    gs00 = gs[ccntr,2].subgridspec(2,1)
    gplt_2a = fig7.add_subplot(gs00[0])
    gplt_2b = fig7.add_subplot(gs00[1])

    if add_to_fig5 == True:

        f5plt_0= fig5.add_subplot(gs5[loc_f5,0])
        
        f5plt_1= fig5.add_subplot(gs5[loc_f5,1::])



    #loop through all q values values
    for j in range(len(q_load_arr)):
        myalpha=0.9
        #mycolor=plot_colors[j]
        mylinestyle = all_linestyles[j]        
        
        
        myfitfile = q_text_base + q_load_arr[j] + '.csv'
        mydf_init,mydf_fit = read_fit_result(os.path.join(input_dir,myfitfile))
        
        full_spc_ids = clean_spc_ids(mydf_fit.spc_id.copy())
        #please_stop_here()
        
        
        mydf_fit_spc = mydf_fit[full_spc_ids == myinputfile]
        
    
        xv  = data_x[rt.find_nearest(data_x,xlims[0]):rt.find_nearest(data_x,xlims[1])]
    
        yv = np.zeros(xv.shape) 
        meas_spc_sub=data_y-data_x*mydf_fit_spc.bkr_s.values-mydf_fit_spc.bkr_c.values
        meas_spc_sub_smooth = savgol_filter(meas_spc_sub,21,4)
        if j == 0:
            gplt_0.plot(data_x,meas_spc_sub,linewidth=2,linestyle=mylinestyle,color='k',alpha=0.9)
            gplt_1.plot(data_x,meas_spc_sub_smooth,linewidth=1,linestyle=mylinestyle,color='k',alpha=0.9)
            gplt_1_alt.plot([],[],linewidth=1,linestyle=mylinestyle,color='k',alpha=0.9)
            
            if add_to_fig5 == True:
                if loc_f5==lowest_f5_ind:
                    print("ADDING...",i)
                    f5plt_0.plot([], [], color='w', alpha=0, label='A-D:')
                    f5plt_1.plot([],[], color='w', alpha=0, label='E-G:')
                    
                f5plt_0.plot(data_x,meas_spc_sub,linewidth=2,linestyle=mylinestyle,color='k',alpha=0.9,label = 'Meas')
                
                f5plt_1.plot(data_x,meas_spc_sub_smooth,linewidth=1,linestyle=mylinestyle,color='k',alpha=0.9,label='Sm(Meas)')
                if loc_f5==lowest_f5_ind:
                    
                    f5plt_1.plot(np.zeros(1), np.zeros([1,3]), color='w', alpha=0, label=' ')
                    #f5plt_1.plot(np.zeros(1), np.zeros([1,3]), color='w', alpha=0, label=' ')
                    #f5plt_1.plot(np.zeros(1), np.zeros([1,3]), color='w', alpha=0, label=' ')
          
        
        
        for index,mypeak in mydf_init.iterrows():
            pk_nm = mypeak.pk_name
            if (mypeak.pk_type == 'Lorentzian'):
                pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values
                pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values
                pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values
                pk_h = amplitude_to_height(pk_a,pk_s,curve='Lorentzian')
                pk_m = lorentzian(xv,pk_c,pk_s,pk_a)
            elif (mypeak.pk_type == 'Gaussian'):
                pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values
                pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values
                pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values
                pk_m = gaussian(xv,pk_c,pk_s,pk_a)
                
            if (pk_nm == 'd1'):
                d1_c[j,i] = pk_c
                d1_s[j,i] = pk_s
                d1_a[j,i] = pk_a
                if j == 0:
                    d1_base_c = pk_c
                    d1_base_s = pk_s
                    d1_base_a = pk_a
            if (pk_nm == 'g'):
                g_c[j,i] = pk_c
                g_s[j,i] = pk_s
                g_a[j,i] = pk_a
                g_h[j,i] = pk_h
                if j == 0:
                    g_base_c = pk_c
                    g_base_s = pk_s
                    g_base_a = pk_a              
            if (pk_nm == 'd2'):
                d2_c[j,i] = pk_c
                d2_s[j,i] = pk_s
                d2_a[j,i] = pk_a
                d2_h[j,i] = pk_h
                if j == 0:
                    d2_base_c = pk_c
                    d2_base_s = pk_s
                    d2_base_a = pk_a                     
                        
                
                
            gplt_1.plot(xv,pk_m,linewidth=1,linestyle=mylinestyle,color=peak_colors[index],alpha=0.8)
            if add_to_fig5 == True:
                f5plt_1.plot(xv,pk_m,linewidth=1,linestyle=mylinestyle,color=peak_colors[index],alpha=0.8,label=j_name[j] +'-'+ pk_nm)
                
            if j == 0:
                gplt_1_alt.plot([],[],linewidth=1,linestyle=mylinestyle,color=peak_colors[index],alpha=0.8)

                
                
            yv = yv + pk_m  
        gplt_0.plot(xv,yv,linewidth=2,linestyle=mylinestyle,color=plot_colors[j],alpha=0.8)
        
        gplt_2b.scatter(g_h[j,i]/d2_h[j,i],1,color=plot_colors[j],alpha=0.8,edgecolor='k',marker=init_markers[j])
        gplt_2a.scatter(d1_c[j,i],1,color=plot_colors[j],alpha=0.8,edgecolor='k',marker=init_markers[j])
        gplt_1_alt.plot([],[],color=plot_colors[j],linestyle = all_linestyles[j])
        gplt_1_alt.scatter([],[],color=plot_colors[j],edgecolor='k',marker=init_markers[j])
        
        if add_to_fig5 == True:
            f5plt_0.plot(xv,yv,linewidth=2,linestyle=mylinestyle,color=plot_colors[j],alpha=0.8,label=j_name[j])
           
            
        #print(f"{j}:{q_load_arr[j]},{q_init_val[j]},{q_final_val[j]}")
            #myax.plot(xv,pk_m,color=color_range[index],label=pk_nm,linestyle='--')

        
        #.plot([],[],color=plot_colors[j])
        #gplt_1.plot(xv,yv,alpha=myalpha,color=mycolor,linewidth=2,linestyle=mylinestyle)
        #gplt_0.plot(xv,yv,alpha=myalpha,color=mycolor,linewidth=2,linestyle='--')        
    my_gd2_rat = g_h[:,i]/d2_h[:,i]   
    my_d1 = d1_c[:,i]     

    
        
    minlim1=0.995
    minlim2=0.8
    maxlim1 = 1.005
    maxlim2 = 1.2
    d1a_min = np.nanmin(np.delete(d1_a[:,i],0))*minlim2
    d1a_max = np.nanmax(np.delete(d1_a[:,i],0))*maxlim2
    d1s_min = np.nanmin(np.delete(d1_s[:,i],0))*minlim1
    d1s_max = np.nanmax(np.delete(d1_s[:,i],0))*maxlim1   
    d1c_min = np.nanmin(np.delete(d1_c[:,i],0))*minlim1
    d1c_max = np.nanmax(np.delete(d1_c[:,i],0))*maxlim1   
    ga_min = np.nanmin(np.delete(g_a[:,i],0))*minlim2
    ga_max = np.nanmax(np.delete(g_a[:,i],0))*maxlim2
    gs_min = np.nanmin(np.delete(g_s[:,i],0))*minlim1
    gs_max = np.nanmax(np.delete(g_s[:,i],0))*maxlim1
    gc_min = np.nanmin(np.delete(g_c[:,i],0))*minlim1
    gc_max = np.nanmax(np.delete(g_c[:,i],0))*maxlim1



    mc='C0'
    #gplt_1.plot(data_x,meas_spc_sub,'k',linewidth=2,zorder=-999)
    #gplt_0.plot(data_x,bkr_e6,color=plot_colors[0])
    #gplt_0.plot(data_x,bkr_e5,color=plot_colors[1])
    #gplt_0.plot(data_x,bkr_e4,color=plot_colors[2])
    #jx,jy = add_jitter(d1_c[:,i])
    #gplt_2a.scatter(d1_c[:,i],np.ones(d1_c[:,i].shape),color=plot_colors,alpha=0.8,edgecolor='k')
    #gplt_2b.scatter(g_h[:,i]/d2_h[:,i],np.ones(g_c[:,i].shape),c=plot_colors,edgecolor='k',s=init_markers)
    
    #gplt_1.set_ylim([0,maxv])
    gplt_0.set_xlim(xlims)
    gplt_1.set_xlim(xlims)
    gplt_1.set_yticks([])
    
    if add_to_fig5 == True:
        
        f5plt_0.set_xticks(f5A_xlab)# = [1000,1330,1600]
        f5plt_1.set_xticks(f5E_xlab)#f5E_xlab = [1000,1200,1400,1600]
        
        f5plt_0.set_xlim(xlims)
        f5plt_1.set_xlim(xlims)
        f5plt_0.set_ylim([0,1.2*np.nanmax(meas_spc_sub[ind_s:ind_e])])
        f5plt_1.set_ylim([0,1.2*np.nanmax(meas_spc_sub[ind_s:ind_e])])
        f5plt_0.text(0.05,0.9,f5labels0[loc_f5],transform = f5plt_0.transAxes)
        f5plt_1.text(0.05,0.9,f5labels1[loc_f5],transform = f5plt_1.transAxes)
        
        f5plt_1.yaxis.set_label_position("right")
        f5plt_1.yaxis.tick_right()
        
        if loc_f5==0:

            #
            f5plt_0.text(0.29,0.9,"Data/Sum Fits",transform = f5plt_0.transAxes)
            f5plt_1.text(0.37,0.9,"Final Fitted Peaks",transform = f5plt_1.transAxes)
            #please_stop_here()
        
        if loc_f5 == 2:
            f5plt_1.set_ylabel('                     Raman Intensity \ Arb. Units') 
            f5plt_0.set_ylabel('                     Raman Intensity \ Arb. Units')               
        
        if loc_f5 < 3:
            f5plt_0.set_xticklabels('')
            f5plt_1.set_xticklabels('')
        else:
            f5plt_0.legend(bbox_to_anchor=(-0.32, -0.3),ncol=6,loc='upper left')
            
            #f5plt_1.plot(np.zeros(1), np.zeros([1,3]), color='w', alpha=0, label=' ')
            

            ax_han,ax_lab = f5plt_1.get_legend_handles_labels()
            ax_lab = [mylab.replace("d","D").replace("g","G") for mylab in ax_lab]
            ax_lab[0] = 'E-G:'     
            #order = list(range(1,len(ax_han)))
            #order.append(0)
            
            #f5plt_1.legend(bbox_to_anchor=(-0.67, -0.5),ncol=5,loc='upper left')
            f5plt_1.legend(ax_han,ax_lab,bbox_to_anchor=(-0.68, -0.5),ncol=5,loc='upper left')
            

            
            f5plt_1.set_xlabel('Wavenumber \ cm-1') 
            f5plt_0.set_xlabel('Wavenumber \ cm-1')   
        
        
    gplt_0.set_ylabel("Intensity\n \Arb. Units")
    gplt_1_alt.set_yticklabels('')
    


    if ccntr == 0:
        gplt_0.set_title("Data/Sum Fits")
        gplt_1.set_title("Fit Peaks")
        gplt_2a.set_title("Fitted Variability")
        gplt_1_alt.legend(q_plot_legend,bbox_to_anchor=(0.5, 1.4),ncol=5,loc='center')
        gplt_0.set_xticklabels('')
        gplt_1.set_xticklabels('')
        gplt_1.set_yticklabels('')
    #if fig7_cntr  != 1:
        #gplt_2a.set_xticklabels('')

    elif ccntr < plot_n_rows:
        #gplt_2b.set_xticklabels('')
        gplt_0.set_xticklabels('')
        gplt_1.set_xticklabels('')
        gplt_1.set_yticklabels('')
    else:
        gplt_1.set_yticklabels('')
        gplt_1.set_xlabel('Wavenumber \ cm-1') 
        gplt_0.set_xlabel('Wavenumber \ cm-1') 
        gplt_2b.set_xlabel('Wavenumber \ cm-1')

        
          
    
    
    gplt_0.text(0.05,0.9,mylabels[0],transform = gplt_0.transAxes)
    gplt_1.text(0.05,0.9,mylabels[1],transform = gplt_1.transAxes)
    gplt_2a.text(0.05,0.8,mylabels[2],transform = gplt_2a.transAxes)
    gplt_2b.text(0.05,0.8,mylabels[3],transform = gplt_2b.transAxes)
    gplt_2a.text(0.95,0.8, "D1 Center",horizontalalignment='right',transform = gplt_2a.transAxes)
    gplt_2b.text(0.99,0.8,"G/D2 Intensity Ratio",horizontalalignment='right',transform = gplt_2b.transAxes)
    
    
    
    print(f'{plot_names[tncntr]} - {mylabels[0]}')
    print(f'G/D2Range:{np.nanmin(my_gd2_rat)}-{np.nanmax(my_gd2_rat)} | {np.nanmax(my_gd2_rat)-np.nanmin(my_gd2_rat)}  ')    
    print(f'D1Range:{np.nanmin(my_d1)}-{np.nanmax(my_d1)}')     
           
    
    gplt_0.set_ylim([0,1.2*np.nanmax(meas_spc_sub[ind_s:ind_e])])
    gplt_1.set_ylim([0,1.2*np.nanmax(meas_spc_sub[ind_s:ind_e])])
    gplt_2a.set_xlim([1300,1360])
    
    gplt_2a.tick_params(top=False,bottom=True,right=False,left=False)#,labelbottom=False,labeltop=True,right=True,left=False,labelleft=False,labelright=True)
    gplt_2b.tick_params(top=False,bottom=True,right=False,left=False)
    #gplt_2a.set_xticks([1300,1340,1380])
    gplt_2a.set_xticks([1320,1340])
    #gplt_2b.set_xticks([1580,1600])
    maxrat = np.ma.masked_invalid(g_h[:,i]/d2_h[:,i]).max()
    print(g_h[:,i]/d2_h[:,i])

    if maxrat <=2:
        gplt_2b.set_xlim([-0.1,2.1])
        gplt_2b.set_xticks([0,1,2])
    else:
        gplt_2b.set_xlim([-0.1,maxrat*1.1])
        gplt_2b.set_xticks([0,np.round(0.66*maxrat)])
               
    #gplt_2a.set_ylabel("D")
    #gplt_2b.set_ylabel("G")#,color=mc)
    #gplt_2a.set_ylabel("D")
    #gplt_2b.set_ylabel("G")#,color=mc)
    gplt_0.set_ylabel("Intensity\n \Arb. Units")
    gplt_1_alt.set_yticklabels('')
    


    if ccntr == 0:
        gplt_0.set_title("Data/Sum Fits")
        gplt_1.set_title("Fit Peaks")
        gplt_2a.set_title("Fitted Variability")
        gplt_1_alt.legend(q_plot_legend,bbox_to_anchor=(0.5, 1.4),ncol=5,loc='center')
        gplt_0.set_xticklabels('')
        gplt_1.set_xticklabels('')
        gplt_1.set_yticklabels('')
    #if fig7_cntr  != 1:
        #gplt_2a.set_xticklabels('')

    elif ccntr < plot_n_rows:
        #gplt_2b.set_xticklabels('')
        gplt_0.set_xticklabels('')
        gplt_1.set_xticklabels('')
        gplt_1.set_yticklabels('')
    else:
        gplt_1.set_yticklabels('')
        gplt_1.set_xlabel('Wavenumber \ cm-1') 
        gplt_0.set_xlabel('Wavenumber \ cm-1') 
        gplt_2b.set_xlabel('Wavenumber \ cm-1')
        
 
    gplt_2a.set_yticks([])
    gplt_2b.set_yticks([])
    gplt_2a.tick_params(axis="x",direction="in", pad=-15)
    gplt_2b.tick_params(axis="x",direction="in", pad=-15)
    
    ccntr = ccntr + 1

    if ccntr > plot_n_rows:
        savename = plot_names[tncntr]
        plt.savefig(savename)
        ccntr=0
        tncntr=tncntr+1
        #if tncntr==1:
        #    break
        plt.close(fig7)
        fig7 = plt.figure(figsize=[7.5,10])
        gs = fig7.add_gridspec(5,3,wspace=0.05,hspace=0.1)    
        
#fig5.savefig("Figure5.png")    
    


