   # -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:53:35 2024

@author: david.c.doughty.civ
"""

#Generate initial figures

#Here we generate initial figures. 

#Initial figures

#One key note - changing the order did not affect the outputs - i.e. as you would hope the lumincence removal didn't change the numbers/values. 

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rs_tools as rt
rt.plot_format(10)
import os
import matplotlib.cm
import matplotlib.patches as patches
import matplotlib.colors as colors
import fit_spc as ft



def calcnoise(measured,modeled):
    #meas_data_shrt-yv
    residual = measured-modeled
    resi_mean = np.nanmean(residual)
    residual_adj = residual-resi_mean
    noise = np.sqrt(np.nansum(residual_adj**2)/len(residual))
    return noise

            

def change_yax_pos(ax):
    vlen,hlen = ax.shape
    rightax = hlen-1
    for i in range(vlen):
        ax[i,rightax].yaxis.set_label_position("right")
        ax[i,rightax].yaxis.tick_right()
        
def calc_snr(my_pkvals,my_meas_data,my_pk_ctr=None,my_pk_sig = None):
    #
    #   Function to calculate SNR values for codes. 
    #   Inputs: my_pkvals: nxm array of individual peak values where n = number of peaks
    #           my_meas_data: array of length "m" with measured data
    #           my_peak_ctr: optional - contains the peak centers. May be used for different implementation
    #           my_pk_sig: optional - not used but may be used in the future. 
    #
    npks,nvars = my_pkvals.shape
    my_pk_sum = np.nansum(my_pkvals,axis=0)
    my_noise = calcnoise(my_meas_data,my_pk_sum)
    my_snr = np.full(npks,np.nan)
    
    
    for k in range(npks):
        max_pk = np.nanmax(my_pkvals[k,:])
        my_snr[k] = max_pk/my_noise
        
    return my_snr
        
xlims=[900,1800]
#Indices for the ALS fit
ind_s = 133
ind_e = 647

cmap = matplotlib.colormaps['jet']



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


input_directory = 'data_files'
for i in range(len(all_spc_ids)):

    myinputfile = os.path.join(input_directory,all_spc_ids[i])
    mydata = np.loadtxt(myinputfile,delimiter=' ',skiprows=0)
    data_x = mydata[:,0]
    data_y = mydata[:,1]
    
    print(f'{all_spc_ids[i]}:213/544:{data_x[213]}-{data_x[544]}|241/511:{data_x[241]}-{data_x[511]}')


mylabel=['A.','B.','C.','D.','E.']

 
encoding = 'utf-8'

with open("Table1_v1a.csv",'w',encoding=encoding) as fh: 
    fh.write(u"Spc #,FID,Fig,2L-RMSE,2LG-RMSE,BWFL-RMSE,2L-D1,2LG-D1,BWFL-D1,2L-G,LG-G,BWFL-G,LG-D3,BWFL-q,SNR-2L-G,SNR-LG-G,SNR-BWFL-G,SNR-2L-D,SNR-LG-D,SNR-BWFL-D,\n")

#\u03C7 #This works
#1D712
#1D712
########################################
#
#   Set up first set of spectra
#
#


spcfitfile = "2L1G_spc_updated.csv"
with open(spcfitfile,'w') as fh:    
    fh.write('')  

l2_dir = 'fit_results'

l2_load_arr = [
        '-daga',
        '-dagb',
        '-dagc',
        '-dbga',
        '-dbgb',
        '-dbgc',
        '-dcga',
        '-dcgb',
        '-dcgc']

l2_base = 4

nvars_l2 = len(l2_load_arr)

l2_text_base = 'dg_2pk_lor-apprx'

d1_init_pos_data = [1310,1310,1310,1330,1330,1330,1350,1350,1350]
g_init_pos_data =  [1580,1600,1620,1580,1600,1620,1580,1600,1620]



d_final_val_all = np.full((len(all_spc_ids),len(d1_init_pos_data)),np.nan)
d_init_val_all = np.full((len(all_spc_ids),len(d1_init_pos_data)),np.nan)

g_final_val_all = np.full((len(all_spc_ids),len(d1_init_pos_data)),np.nan)
g_init_val_all = np.full((len(all_spc_ids),len(d1_init_pos_data)),np.nan)

###############################################
#
#   2 Lorentzian 1 Gaussian Setup
#
#

d3_load_arr = [
        'p1480h05',
        'p1480h25',
        'p1480h50',
        'p1500h05',
        'p1500h25',
        'p1500h50',
        'p1520h05',
        'p1520h25',
        'p1520h50']

d3_base = 4

d3_text_base = 'dg_3pk_smp_ratapprx-'
d3_dir = 'fit_results'
d3_init_pos_data = [1460,1460,1460,1500,1500,1500,1540,1540,1540]
d3_init_h_data = [0,0.15,0.30,0,0.15,0.30,0,0.15,0.30]
d3_init_h_val= [0,0.15,0.30,0,0.15,0.30,0,0.15,0.30] #Read from the 

d3d1_c = np.full((len(d3_load_arr),len(all_spc_ids)),np.nan)
d3d1_s = np.full((len(d3_load_arr),len(all_spc_ids)),np.nan)
d3d1_a = np.full((len(d3_load_arr),len(all_spc_ids)),np.nan)

d3g_c = np.full((len(d3_load_arr),len(all_spc_ids)),np.nan)
d3g_s = np.full((len(d3_load_arr),len(all_spc_ids)),np.nan)
d3g_a = np.full((len(d3_load_arr),len(all_spc_ids)),np.nan)

d3d3_c = np.full((len(d3_load_arr),len(all_spc_ids)),np.nan)
d3d3_s = np.full((len(d3_load_arr),len(all_spc_ids)),np.nan)
d3d3_a = np.full((len(d3_load_arr),len(all_spc_ids)),np.nan)

###############################################
#
#   BWF fits
#
#

q_load_arr = [
        '-q2',        
        '-q3',        
        '-q4',        
        '-q5',
        '-q6',
        '-q7',
        '-q8',
        '-q9',
        '',
        '-q11',
        '-q12',
        '-q13',
        '-q14',
        '-q15',
        '-q16',        
        '-q18',
        '-q30',
        '-q40',
        '-q50',
        '-q100',
        '-q500',
        '-q1000',
        '-q10000']

q_base = 5

q_text_base = 'dg_2pk_bwflor-apprx'

q_init_val_data = [-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-18,-30,-40,-50,-100,-500,-1000,-10000]
#q_init_val_data = [1,10,100,1000,10000,100000,-1,-5,-10,-50,-100,-500,-1000,-10000,-100000]
#q_init_val_data = [1,10,100,1000,10000,100000,-1,-5,-10,-50,-100,-500,-1000,-10000,-100000]
#q_dir = 'j-qvar'
qplottxt = "FigS5_alldata.png"
qplottxtz = "FigS6_alldata_zoom.png"
q_dir = 'fit_results'
#

d3d1_c = np.full((len(d3_load_arr),len(all_spc_ids)),np.nan)

q_final_val = np.full((len(q_init_val_data),len(all_spc_ids)),np.nan)
q_init_val = np.full((len(q_init_val_data),len(all_spc_ids)),np.nan)
q_chisq = np.full((len(q_init_val_data),len(all_spc_ids)),np.nan)
q_rmse_all = np.full((len(q_init_val_data),len(all_spc_ids)),np.nan)

##############################
#
#   Set up second set of spectra
#
#
plot_n_rows = 5 
f,ax = plt.subplots(nrows=plot_n_rows,ncols=1,figsize=[7.5,10],sharex=True,dpi=300)
ncntr=0
myplotnum=1

ftoc1,axtoc1 = plt.subplots(figsize=[7.5,10])
ftoc2,axtoc2 = plt.subplots(figsize=[7.5,10])


for i in range(len(all_spc_ids)):
    
    #####
    #
    #   Handle 2 lor
    #
    d_final_val = np.full(len(d1_init_pos_data),np.nan)
    d_init_val = np.full(len(d1_init_pos_data),np.nan)

    g_final_val = np.full(len(d1_init_pos_data),np.nan)
    g_init_val = np.full(len(d1_init_pos_data),np.nan)

    
    myax = ax[ncntr]
    

    myinputfile = os.path.join(input_directory,all_spc_ids[i])
    print(myinputfile)
    mydata = np.loadtxt(myinputfile,delimiter=' ',skiprows=0)
    data_x = mydata[:,0]
    data_y = mydata[:,1]
    x_min_ind = rt.find_nearest(data_x,xlims[0])
    x_max_ind = rt.find_nearest(data_x,xlims[1])+1      
    xv = data_x[x_min_ind:x_max_ind]

    #### - add extra plot
    myax_a = myax.twinx()
    lab0, = myax_a.plot(data_x,data_y,color='k',alpha=0.5)
    myax_a.set_ylim([0,1.1*np.nanmax(data_y[x_min_ind:x_max_ind])])
    
    myax_a.legend([lab0],['raw'],bbox_to_anchor=(0.99,0.85),loc='right')
    #loop through all peak position values 
    for j in range(len(l2_load_arr)):
        myfitfile = l2_text_base + l2_load_arr[j] + '.csv'
        myfitpath= os.path.join(l2_dir,myfitfile)
        mydf_init,mydf_fit = ft.read_fit_result(myfitpath)
        
        mydf_fit_spc = mydf_fit[mydf_fit.spc_id == all_spc_ids[i]]
        

        yv = np.zeros(xv.shape) 
        
        if j == 0:
            meas_spc_sub=data_y-data_x*mydf_fit_spc.bkr_s.values.item()-mydf_fit_spc.bkr_c.values.item()
            maxv = np.nanmax(meas_spc_sub[ind_s:ind_e])      
            meas_data_shrt = meas_spc_sub[x_min_ind:x_max_ind]
            lab1,=myax.plot(data_x,meas_spc_sub,'k',linewidth=2)#,alpha=0.5)
        

        d_init_val_all[i,j]  = mydf_init.iloc[1]['pk_center']    
        d_final_val_all[i,j] = mydf_fit_spc['d1_center'].values.item()
        
        g_init_val_all[i,j]  =  mydf_init.iloc[0]['pk_center']
        g_final_val_all[i,j]  = mydf_fit_spc['g_center'].values.item()
        
        npks = len(mydf_init)
        pkvals = np.full((npks,len(xv)),np.nan)
        for index,mypeak in mydf_init.iterrows():
            pk_nm = mypeak.pk_name
            if ((mypeak.pk_type == 'Lorentzian')):
                pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values.item()
                pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values.item()
                pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values.item()
                pk_m = ft.lorentzian(xv,pk_c,pk_s,pk_a)
            pkvals[index,:] = pk_m
            if i == 1:
                axtoc1.plot(xv,pk_m,alpha=0.5,color='C0',linewidth=1)
            yv = yv + pk_m
            
            if ((j == l2_base) & ('d1' in pk_nm)):
                l2_d1c = pk_c   
                pkind_d1 = rt.find_nearest(xv,l2_d1c)
            elif ((j == l2_base) & ('g' in pk_nm)):
                l2_gc = pk_c 
                pkind_g = rt.find_nearest(xv,l2_gc)
   
        if j == l2_base:
           l2_redchisq = mydf_fit_spc.chisq_red.values.item() 
           l2_rmse =  np.sqrt(mydf_fit_spc.mse.values.item())
           my_snr = calc_snr(pkvals,meas_data_shrt)
           l2_snr_g = my_snr[0]
           l2_snr_d = my_snr[1]
    
        print(f"{j}:{l2_load_arr[j]},{d_init_val_all[i,j]},{d_final_val_all[i,j]},{g_init_val_all[i,j]},{g_final_val_all[i,j]}")
            #myax.plot(xv,pk_m,color=color_range[index],label=pk_nm,linestyle='--')
        myalpha=0.5
        myax.plot(xv,yv,alpha=myalpha,color='C0',linewidth=2)
        
        #Added for TOC art

 
    #####
    #
    #   Handle 2 lor 1 gaussian
    #        

    for j in range(len(d3_load_arr)):
        yv = np.zeros(xv.shape) 
            
        myfitfile = d3_text_base + d3_load_arr[j] + '.csv'
        myfitpath= os.path.join(d3_dir,myfitfile)
        
        mydf_init,mydf_fit = ft.read_fit_result(myfitpath)
        #import pdb
        #pdb.set_trace()
        
        mydf_fit_spc = mydf_fit[mydf_fit.spc_id == all_spc_ids[i]]
        
        #xv = np.arange(xlims[0],xlims[1])
        #yv = np.zeros(xv.shape) 
        
        #d3_init_pos[j] = mydf_init.iloc[2]['pk_center']
        #d3_final_pos[j] = mydf_fit_spc['d3_center']
        #d3_init_a[j] = mydf_init.iloc[2]['pk_amplitude']
        #d3_final_a[j] = mydf_fit_spc['d3_amplitude']
        

            
        npks = len(mydf_init)
        pkvals = np.full((npks,len(xv)),np.nan)     
        
        for index,mypeak in mydf_init.iterrows():
            pk_nm = mypeak.pk_name
            if (mypeak.pk_type == 'Lorentzian'):
                pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values.item()
                pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values.item()
                pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values.item()
                pk_m = ft.lorentzian(xv,pk_c,pk_s,pk_a)
            elif (mypeak.pk_type == 'Gaussian'):
                pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values.item()
                pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values.item()
                pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values.item()
                pk_m = ft.gaussian(xv,pk_c,pk_s,pk_a)
  
            pkvals[index,:] = pk_m
            if j == d3_base:
                d3_redchisq = mydf_fit_spc.chisq_red.values.item() 
            if (pk_nm == 'd1'):
                d3d1_c[j,i] = pk_c
                d3d1_s[j,i] = pk_s
                d3d1_a[j,i] = pk_a
                if j == d3_base:
                    d3d1_base_c = pk_c
                    d3d1_base_s = pk_s
                    d3d1_base_a = pk_a
                    pkind_d3d1 = rt.find_nearest(xv,d3d1_base_c)
            if (pk_nm == 'g'):
                d3g_c[j,i] = pk_c
                d3g_s[j,i] = pk_s
                d3g_a[j,i] = pk_a
                if j == d3_base:
                    d3g_base_c = pk_c
                    d3g_base_s = pk_s
                    d3g_base_a = pk_a 
                    pkind_d3g= rt.find_nearest(xv,d3g_base_c)
            if (pk_nm == 'd3'):
                d3d3_c[j,i] = pk_c
                d3d3_s[j,i] = pk_s
                d3d3_a[j,i] = pk_a
                 
                if j == d3_base:
                    d3d3_base_c = pk_c
                    d3d3_base_s = pk_s
                    d3d3_base_a = pk_a  
                    pkind_d3d3= rt.find_nearest(xv,d3g_base_c)
    

            yv = yv + pk_m    
        #print(f"{j}:{q_load_arr[j]},{d3_init_pos[j]},{d3_final_pos[j]},{d3_init_a[j]},{d3_final_a[j]}")
            #myax.plot(xv,pk_m,color=color_range[index],label=pk_nm,linestyle='--')
        myalpha=0.7
        if j == d3_base:
            d3_redchisq = mydf_fit_spc.chisq_red.values.item()     
            d3_rmse = np.sqrt(mydf_fit_spc.mse.values.item())
            my_snr = calc_snr(pkvals,meas_data_shrt)
            d3_snr_g = my_snr[0]
            d3_snr_d = my_snr[1]
  
        if 'h0' in d3_load_arr[j]:
            mylinestyle = ':'
            mycolor = 'C2'
            myax.plot(xv,yv,alpha=myalpha,color=mycolor,linestyle=mylinestyle,linewidth=3)
        else:
            mylinestyle='--'
            mycolor='C1'
            myax.plot(xv,yv,alpha=myalpha,color=mycolor,linestyle=mylinestyle,linewidth=2)
            
        figdatatxt = f"{all_spc_ids[i]}\t\t{d3_load_arr[j]}\t\t\t{mydf_init.iloc[2].pk_center:.4e},{mydf_init.iloc[2].pk_amplitude:.3e}\t{d3g_c[j,i]:.4e}\t{d3d1_c[j,i]:.6e}\t{d3d3_c[j,i]:.4e}\t{d3d3_a[j,i]:.3e}\t{d3g_a[j,i]:.3e}\t{mydf_fit_spc.chisq_red.values.item() :.1f}\t{mydf_fit_spc.mse.values.item():.1f}\n"
        spcfitfile = "2L1G_spc_updated.csv"
        with open(spcfitfile,'a') as fh:    
            fh.write(figdatatxt)  
    #####
    #
    #   Handle q
    #         
    figdatatxt = f"{all_spc_ids[i]}\n"
    spcfitfile = "BWF_spc.csv"
    with open(spcfitfile,'a') as fh:    
        fh.write(figdatatxt)  
        
    for j in range(len(q_load_arr)):
        yv = np.zeros(xv.shape) 
        myfitfile = q_text_base + q_load_arr[j] + '.csv'
        myfitpath= os.path.join(q_dir,myfitfile)
        mydf_init,mydf_fit = ft.read_fit_result(myfitpath)
        
        mydf_fit_spc = mydf_fit[mydf_fit.spc_id == all_spc_ids[i]]
        
        #xv = np.arange(xlims[0],xlims[1])
        #yv = np.zeros(xv.shape) 
        npks = len(mydf_init)
        pkvals = np.full((npks,len(xv)),np.nan)
        
        for index,mypeak in mydf_init.iterrows():
            pk_nm = mypeak.pk_name
            if (mypeak.pk_type == 'Lorentzian'):
                pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values.item()
                pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values.item()
                pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values.item()
                pk_m = ft.lorentzian(xv,pk_c,pk_s,pk_a)
            elif ((mypeak.pk_type == 'BreitWignerFano')):
                pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values.item()
                pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values.item()
                pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values.item()
                pk_q =  mydf_fit_spc[str(pk_nm+'_q')].values.item()
                pk_m = ft.breitwignerfano(xv,pk_c,pk_s,pk_a,pk_q)   
                q_init_val[j,i] = mydf_init.iloc[index]['pk_par4']
                q_final_val[j,i] = mydf_fit_spc['g_q']
            elif ((mypeak.pk_type == 'BWF_spc')):
                pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values.item()
                pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values.item()
                pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values.item()
                pk_q =  mydf_fit_spc[str(pk_nm+'_q')].values.item()
                pk_m = ft.BWF_spc(xv,pk_c,pk_s,pk_a,pk_q)   
                q_init_val[j,i] = mydf_init.iloc[index]['pk_par4']
                q_final_val[j,i] = mydf_fit_spc['g_q'] .values.item()               

            yv = yv + pk_m   
            pkvals[index,:] = pk_m
            
            if ((j == q_base) & ('d1' in pk_nm)):
                q_d1c = pk_c
                pkind_qd1 = rt.find_nearest(xv,q_d1c)
            elif ((j == q_base) & ('g' in pk_nm)):
                q_base_val = pk_q
                q_gc = pk_c
                pkind_qg = rt.find_nearest(xv,q_gc)                
        if j == q_base:
           q_redchisq_base = mydf_fit_spc.chisq_red.values.item() 
           q_rmse = np.sqrt(mydf_fit_spc.mse.values.item())
           my_snr = calc_snr(pkvals,meas_data_shrt)
           q_snr_g = my_snr[0]
           q_snr_d = my_snr[1]
        q_rmse_all[j,i] = np.sqrt(mydf_fit_spc.mse.values.item()) 
        q_chisq[j,i] = mydf_fit_spc.chisq_red.values.item() 
        #pdb.set_trace()
        
        figdatatxt = f"{q_load_arr[j]}\t\t\t{q_init_val[j,i]:.1e}\t{q_final_val[j,i]:.1e}\t{ q_chisq[j,i]:.1f}\n"
        spcfitfile = "BWF_spc.csv"
        with open(spcfitfile,'a') as fh:    
            fh.write(figdatatxt)  
       
        #print(f"{j}:{q_load_arr[j]},{q_init_val[j]},{q_final_val[j]}")
            #myax.plot(xv,pk_m,color=color_range[index],label=pk_nm,linestyle='--')
        
        if (q_init_val[j,i] < 0) and (q_init_val[j,i] > -10):
            #mycolor = cmap(np.log10(np.abs(q_final_val[j]))/q_max)
            
            mycolor='C3'
            mylinestyle = '--'

        else:
            mycolor='C4'
            mylinestyle = ':'
        myax.plot(xv,yv,alpha=myalpha,color=mycolor,linewidth=2,linestyle=mylinestyle)    
 
    
    #####
    #
    #   End of section!!
    #
 
    
    #Finish plot

    myax.set_ylim([0,1.1*maxv])
    myax.set_xlim(xlims)
    
    spctxt = mylabel[ncntr] + f'Spectrum {i+1}'
    myax.text(0.02,0.9,spctxt,transform=myax.transAxes)
    myax.xaxis.set_tick_params(direction='in',which='both')
    
    
    
    #Output Table 1 data
    file_id = all_spc_ids[i].replace("selected_","").replace(".txt","")
    figtxt = f"Fig{myplotnum}{mylabel[ncntr]}"
    #tabletxt = f"{i+1},{file_id}, {figtxt},{l2_redchisq:.1f} ,{d3_redchisq:.1f}, {q_redchisq_base:.1f}, {l2_d1c:.1f},{d3d1_base_c},{q_d1c:.1f},{l2_gc:.1f},{d3g_base_c},{q_gc:.1f},{d3d3_base_c},{q_base_val:.1f},{l2_snr_g:.1f},{d3_snr_g:.1f},{q_snr_g:.1f},{l2_snr_d:.1f},{d3_snr_d:.1f},{q_snr_d:.1f}\n"
    tabletxt = f"{i+1},{file_id}, {figtxt},{l2_rmse:.1f} ,{d3_rmse:.1f}, {q_rmse:.1f}, {l2_d1c:.1f},{d3d1_base_c},{q_d1c:.1f},{l2_gc:.1f},{d3g_base_c},{q_gc:.1f},{d3d3_base_c},{q_base_val:.1f},{l2_snr_g:.1f},{d3_snr_g:.1f},{q_snr_g:.1f},{l2_snr_d:.1f},{d3_snr_d:.1f},{q_snr_d:.1f}\n"
   
    with open("Table1_v1a.csv",'a') as fh:    
        fh.write(tabletxt)  
     
            
    ncntr=ncntr+1
    if ncntr > plot_n_rows-1:      
        lab2, = ax[0].plot([],[],'C0')
        lab3, = ax[0].plot([],[],'C1',linestyle = '--')
        lab4, = ax[0].plot([],[],'C2',linestyle=':',linewidth=3)      
        lab5, = ax[0].plot([],[],'C3',linestyle = '--')
        lab6, = ax[0].plot([],[],'C4',linestyle = ':')
        ax[0].legend([lab1,lab2,lab3,lab4,lab5,lab6],['Measured','2 Lorentzian',r'2 Lor 1 Gaus $I_{init}$>0.05$I^{meas,sm}_{\omega_0}$',r'2 Lor 1 Gaus $I_{init}$=0.05$I^{meas,sm}_{\omega_0}$','BWF (-10<q$_{init}$ <0)','Other BWF q$_{init}$'],bbox_to_anchor=(0.5,1.2),ncol=3,loc='center')
        
        f.subplots_adjust(hspace=0.05)
        ax[2].set_ylabel(r'Intensity \ Arb. Units')
        myax.set_xlabel(r'Wavenumber\ cm-1')
        #f.savefig(f'Fig_{myplotnum}.png')
        f.savefig(f'Fig_S{myplotnum+1}_zoomed.png')
        #please_stop_here()
        myax.set_xlim([400,3100])
        f.savefig(f'Fig_S{myplotnum+1}.png')
        myplotnum=myplotnum+1
        plt.close(f)
        ncntr=0
        f,ax = plt.subplots(nrows=5,ncols=1,figsize=[7.5,10],sharex=True,dpi=300)
 
                
 

###############################################################################
#
#
#   Generate Figure 4
#
#


useind=5

d1_dc = np.subtract(d3d1_c,d3d1_c[useind,:])
d3_dc = np.subtract(d3d3_c,d3d3_c[useind,:]) 
g_dc = np.subtract(d3g_c,d3g_c[useind,:])

delta_max = 20
delta_min = -20
dm = 3
d1_dc[d1_dc > delta_max] = delta_max
d3_dc[d3_dc > delta_max] = delta_max
g_dc[g_dc > delta_max] = delta_max

yticks = [-20,-10,0,10,20]
ytick_labels = ['<-20','-10','0','10','>20']

d1_dc[d1_dc < delta_min] = delta_min
d3_dc[d3_dc < delta_min] = delta_min
g_dc[g_dc < delta_min] = delta_min


print(np.nanmean(d1_dc,axis=0))
       
xlims=[800,2000]
q_load_arr = [
        'p1480h05' ,
        'p1480h25',
        'p1480h50',
        'p1500h05' ,
        'p1500h25',
        'p1500h50',
        'p1520h05' ,
        'p1520h25',
        'p1520h50']

q_text_base = 'dg_3pk_smp_ratapprx-'

marker = ['<','s','*','>','p','X','^','o','d']
plot_colors = ['cyan','C1','C2','C3','C4','C5','C6','C8','C9']

myxticks = [0,4,9,14]
myxticklabs = ['1','5','10','15']
d3_init_pos_data = [1460,1460,1460,1500,1500,1500,1540,1540,1540]
d3_init_h_data = [0.05,0.25,0.50,0.05,0.25,0.50,0.05,0.25,0.50]
d3_init_h_val= [0.05,0.25,0.50,0.05,0.25,0.50,0.05,0.25,0.50] #Read from the 

rectcolor = [0.5,0.5,0.5]

f,ax = plt.subplots(nrows=1,ncols=3,figsize=(8.5,4))  
nq,ns = d1_dc.shape
for i in range(len(q_load_arr)):
    sv = 0.25-0.5*np.divide(i,nq)
    ax[0].scatter(np.arange(ns)+sv,d1_dc[i,:],marker=marker[i],color=plot_colors[i],alpha=0.7)
    ax[1].scatter(np.arange(ns)+sv,d3_dc[i,:],marker=marker[i],color=plot_colors[i],alpha=0.7)
    ax[2].scatter(np.arange(ns)+sv,g_dc[i,:],marker=marker[i],color=plot_colors[i],alpha=0.7)

ax[1].legend(q_load_arr,bbox_to_anchor=(0.5, 1.075),ncol=5,loc='center')
ax[0].set_ylabel(r"$\Delta$ Peak Center from p1500h50 / cm$^{-1}$")

for myax in ax:
    myax.set_xlabel("Spectrum Number")
    myax.set_ylim([delta_min-dm,delta_max+dm])

    myax.set_yticks(yticks)
    myax.set_yticklabels('')

ax[0].set_yticklabels(ytick_labels)

ax[0].text(0.05,0.9,r"A. D1 ",transform = ax[0].transAxes)
ax[1].text(0.05,0.9,r"B. D3 ",transform = ax[1].transAxes)
ax[2].text(0.05,0.9,r"C. G ",transform = ax[2].transAxes)

ax1a = ax[0].inset_axes(
        [0.1,0.2,0.8,0.2], transform=ax[0].transAxes)
myinputfile = os.path.join('data_files','selected_96919_20160825_001238_0.0_31.0.txt') #Good - narrow peaks
mydata = np.loadtxt(myinputfile,delimiter=' ',skiprows=0)

data_x = mydata[:,0]
data_y = mydata[:,1]
ax1a.plot(data_x,data_y,color='C0')
ax1a.set_xlim([1100,1700])
ax1a.set_yticks([])
ax1a.set_xticks([1200,1600])
ax1a.set_xticklabels(['1200','1600'],fontsize=10,color=rectcolor)
ax1a.set_xlabel(r"Wavenumber \ cm$^{-1}$",fontsize=10,color=rectcolor)

rect1 = patches.Rectangle((0.5,-2),1,4,edgecolor=rectcolor,fill=False,linewidth=2) #facecolor=None)
ax[0].add_patch(rect1)

rect2 = patches.Rectangle((0.5,-21),1,22,edgecolor=rectcolor,fill=False,linewidth=2) #facecolor=None)
ax[1].add_patch(rect2)

for spine in ax1a.spines.values():
    spine.set_edgecolor([0.5,0.5,0.5])
ax[0].set_xticks(myxticks)
ax[0].set_xticklabels(myxticklabs)    
ax[1].set_xticks(myxticks)
ax[1].set_xticklabels(myxticklabs)
ax[2].set_xticks(myxticks)
ax[2].set_xticklabels(myxticklabs)
plt.savefig("FigS10_vard3.png")


###############################################################################
#
#   Generate Q Summary Figure
#
#


mycmap = matplotlib.colormaps['nipy_spectral']


zoom_ylims = [-12,-4]
zoom_xlims = [-21,0]

rect = patches.Rectangle((zoom_xlims[1],zoom_ylims[0]), (zoom_xlims[0]-zoom_xlims[1]),(zoom_ylims[1]-zoom_ylims[0]), linewidth=1, edgecolor='k', facecolor='none')


q_max = 5
f1,ax1 = plt.subplots(figsize=(11,8.5))
my_markers = ['<','>','^','v','1','2','4','h','P','s','*','p','X','o','D']  

minhisq = np.nanmin(q_chisq,axis=0)
nq,nspc = q_chisq.shape

all_handles = []   
for i in range(len(all_spc_ids)):
    #gph = ax1.scatter(q_init_val[:,i],q_final_val[:,i],c=q_redchisq_rat[:,i],cmap=mycmap,vmin=0,vmax=100,marker=my_markers[i],alpha=0.7)
    #gph = ax1.scatter(q_init_val[:,i],q_final_val[:,i],c=q_redchisq_rat[:,i],cmap=mycmap,vmin=0,vmax=100,marker=my_markers[i],alpha=0.7)
    gph = ax1.scatter(q_init_val[:,i],q_final_val[:,i],c=q_rmse_all[:,i],cmap=mycmap,norm=colors.LogNorm(vmin=10,vmax=500),edgecolor='k',marker=my_markers[i],alpha=0.9)
   
    all_handles.append(gph) 
    
ax1.add_patch(rect)
plt.show()
#f1,ax1 = plt.subplots()   
#ax1.set_ylim([-10**9,10**16])   
#ax1.set_xlim([-5*10**7,5*10**7]) 
#ax1.set_xlim([-150,5])   
#axins = ax1.inset_axes([0.9,0.01,0.05,0.9],transform=ax1.transAxes)
cbh = plt.colorbar(gph,orientation="vertical") 
legh = ax1.legend(all_handles,['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'],ncol=4,title='Spectrum Number',loc='lower right')
cbh.set_label(r"RMSE \ Arb. Units.") 
ax1.set_xscale('symlog',linthresh=20) 
ax1.set_yscale('symlog',linthresh=20) 
ax1.set_ylabel(r"$Q_{final}$")
ax1.set_xlabel(r"$Q_{init}$")
plt.savefig(qplottxt)   

ax1.set_xscale('linear') 
ax1.set_yscale('linear') 
ax1.set_ylim([-12,-4])   
ax1.set_xlim([-21,0])   
plt.savefig(qplottxtz) 

#please_stop_here() 


##############################################################################
#
#   Generate figure significance. 
#
#
    
all_spc_ids = np.array (['selected_96919_20160825_001238_0.0_31.0.txt', # Sharp peaks
       'selected_31624_20160825_184739_26.0_32.0.txt' ,# - good one - removed because it's just too clear and not so 'interesting'
       'selected_151_20160825_212530_6.0_19.0.txt'   , #Not clear but sort of rough spectrum
       'selected_44428_20160825_095858_32.0_32.0.txt',#  Weak DGC
       'selected_43052_20160825_180008_22.0_20.0.txt',#Broad DGC
       #'selected_20031_20160825_030638_24.0_11.0.txt',#Luminescence
       'selected_3914_20160825_191905_20.0_42.0.txt' ,
       'selected_12281_20160825_153727_36.0_5.0.txt' ,#Clear hematite
       'selected_63471_20160825_132338_0.0_23.0.txt'#Oxalate
       ])

spectrum_id = ["12","13","11","14","8","15","16","17","18","19"]

##############################
#
#   Set up second set of spectra
#
#

plot_n_rows = 4#5 
plot_n_columns = 2

f,ax = plt.subplots(nrows=plot_n_rows,ncols=plot_n_columns,figsize=[7.5,10],sharex=True)

ncntr=0
ccntr=0
tncntr=0
myplotnum=1
myalpha=0.9
xlims=[900,1800]

mylabel = ['A.','B.','C.','D.','E.','F.','G.','H.','I.','J.']

path_l2 = os.path.join(l2_dir,'dg_2pk_lor-apprx-dbgb.csv')
path_d3 = os.path.join(d3_dir,'dg_3pk_smp_ratapprx-p1500h25.csv')
path_q = os.path.join(q_dir, 'dg_2pk_bwflor-apprx-q7.csv')

q_chisq = np.full((len(q_init_val_data),len(all_spc_ids)),np.nan)
q_rmse_all = np.full((len(q_init_val_data),len(all_spc_ids)),np.nan)



for i in range(len(all_spc_ids)):

    max_g  = np.nan
    max_d = np.nan
    myax = ax[ncntr,ccntr]
    myinputfile = os.path.join(input_directory,all_spc_ids[i])
    print(myinputfile)
    mydata = np.loadtxt(myinputfile,delimiter=' ',skiprows=0)
    data_x = mydata[:,0]
    data_y = mydata[:,1]

    x_min_ind = rt.find_nearest(data_x,xlims[0])
    x_max_ind = rt.find_nearest(data_x,xlims[1])+1
    
    xv = data_x[x_min_ind:x_max_ind] 

    pk_rel_txt = ''
    
    #####
    #
    #   Handle 2 lor
    #
    
    mycolor='C0'
    d_final_val = np.full(len(d1_init_pos_data),np.nan)
    d_init_val = np.full(len(d1_init_pos_data),np.nan)

    g_final_val = np.full(len(d1_init_pos_data),np.nan)
    g_init_val = np.full(len(d1_init_pos_data),np.nan)

    mydf_init,mydf_fit = ft.read_fit_result(path_l2)
    
    mydf_fit_spc = mydf_fit[mydf_fit.spc_id == all_spc_ids[i]]
    
    
    yv = np.zeros(xv.shape) 
    
    #Here we plot the measured values
    meas_spc_sub=data_y-data_x*mydf_fit_spc.bkr_s.values.item()-mydf_fit_spc.bkr_c.values.item()
    meas_data_shrt = meas_spc_sub[x_min_ind:x_max_ind]
    maxv = np.nanmax(meas_spc_sub[ind_s:ind_e])
    myax.set_ylim([0,maxv])
    myax.set_xlim(xlims)
    h_meas, =myax.plot(data_x,meas_spc_sub,'k',linewidth=2)

    
    npks = len(mydf_init)
    pkvals = np.full((npks,len(xv)),np.nan)
    for index,mypeak in mydf_init.iterrows():
        pk_nm = mypeak.pk_name
        if ((mypeak.pk_type == 'Lorentzian')):
            pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values.item()
            pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values.item()
            pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values.item()
            pk_m = ft.lorentzian(xv,pk_c,pk_s,pk_a)

        if  ('d1' in pk_nm):
            l2_d1c = pk_c   
            pkind_d1 = rt.find_nearest(xv,l2_d1c)
            sig_d1 = pk_s
        elif ('g' in pk_nm):
            l2_gc = pk_c  
            pkind_g = rt.find_nearest(xv,l2_gc)
            sig_g = pk_s
   
            #myax.plot(xv,pk_m,alpha=myalpha,color=mycolor,linewidth=1)
            
        yv = yv + pk_m 
        pkvals[index,:] = pk_m

        #myax.plot(xv,pk_m,color=color_range[index],label=pk_nm,linestyle='--')
    noise = calcnoise(meas_data_shrt,yv)
    print(f"L2: {l2_d1c:.1f},{l2_gc:.1f}")
    l2_rmse =  np.sqrt(mydf_fit_spc.mse.values.item())
    l2_redchisq = mydf_fit_spc.chisq_red.values.item()       

    my_snr = calc_snr(pkvals,meas_data_shrt)
    l2_snr_g = my_snr[0]
    l2_snr_d = my_snr[1]
    
    #pk_rel_txt = pk_rel_txt + f'{np.nanmax(pkvals[0,:])/noise:.1f},{np.nanmax(pkvals[1,:])/noise:.1f}\n'
    pk_rel_txt = pk_rel_txt + f'{l2_snr_d:.1f},{l2_snr_g:.1f}\n'
    #myax.plot(xv,yv,alpha=myalpha,color=mycolor,linewidth=2)
    h_l2, = myax.plot(xv,yv,color=mycolor,linewidth=2)
    myax.spines['top'].set_visible(False)
    myax.spines['right'].set_visible(False)
    myax.spines['left'].set_visible(False)
    myax.spines['bottom'].set_visible(False)
    myax.tick_params(axis='both', which='both', bottom=False, top=False,left=False, right=False, labelbottom=False, labelleft=False)
    
    
    #if (i !=4) & (i !=9):
    #
    #    myax.tick_params(axis='both', which='both', bottom=False, top=False,left=False, right=False, labelbottom=False, labelleft=False)
    #else:
    #
    #    myax.tick_params(axis='both', which='both', bottom=True, top=False,left=False, right=False, labelbottom=True,labelleft=False)
    #####
    #
    #   Handle 2 lor 1 gaussian
    #        

    mydf_init,mydf_fit = ft.read_fit_result(path_d3)
    
    mydf_fit_spc = mydf_fit[mydf_fit.spc_id == all_spc_ids[i]]

    yv = np.zeros(xv.shape) 
    mylinestyle='--'
    mycolor='C1'
    npks = len(mydf_init)
    pkvals = np.full((npks,len(xv)),np.nan)
    ind_d1 = np.nan
    ind_g = np.nan
    ind_d3 = np.nan
    ctr_d1 = np.nan
    ctr_g = np.nan
    ctr_d3 = np.nan
    max_d1 = np.nan
    max_g = np.nan
    max_d3 = np.nan
    for index,mypeak in mydf_init.iterrows():
        pk_nm = mypeak.pk_name
        if 'd1' in pk_nm:
            ind_d1 = index
        elif 'g' in pk_nm:
            ind_g = index
        elif 'd3' in pk_nm:
            ind_d3 = index
            
        if (mypeak.pk_type == 'Lorentzian'):
            pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values.item()
            pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values.item()
            pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values.item()
            pk_m = ft.lorentzian(xv,pk_c,pk_s,pk_a)
        elif (mypeak.pk_type == 'Gaussian'):
            pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values.item()
            pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values.item()
            pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values.item()
            pk_m = ft.gaussian(xv,pk_c,pk_s,pk_a)
        elif ((mypeak.pk_type == 'BreitWignerFano')):
            pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values.item()
            pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values.item()
            pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values.item()
            pk_q =  mydf_fit_spc[str(pk_nm+'_q')].values.item()
            pk_m = ft.breitwignerfano(xv,pk_c,pk_s,pk_a,pk_q)
        h_l2g_i, = myax.plot(xv,pk_m,alpha=myalpha,color=mycolor,linewidth=1)
        pkvals[index,:] = pk_m
        if 'd1' in pk_nm:
            ind_d1 = index
            d3ctr_d1 = pk_c
            pkind_d3d1 = rt.find_nearest(xv,d3ctr_d1)
            sig_d1 = pk_s
        elif 'g' in pk_nm:
            ind_g = index
            d3ctr_g = pk_c
            pkind_d3g= rt.find_nearest(xv,d3ctr_g)
            sig_g = pk_s
        elif 'd3' in pk_nm:
            ind_d3 = index
            d3ctr_d3 = pk_c
            pkind_d3d3= rt.find_nearest(xv,d3ctr_d3)
            sig_d3 = pk_s

        yv = yv + pk_m    
        #print(f"{j}:{q_load_arr[j]},{d3_init_pos[j]},{d3_final_pos[j]},{d3_init_a[j]},{d3_final_a[j]}")
            #myax.plot(xv,pk_m,color=color_range[index],label=pk_nm,linestyle='--')
        myalpha=0.7
    h_l2g, = myax.plot(xv,yv,color=mycolor,linewidth=2,linestyle=mylinestyle)     
  
    d3_rmse = np.sqrt(mydf_fit_spc.mse.values.item())
    my_snr = calc_snr(pkvals,meas_data_shrt)
    d3_snr_g = my_snr[0]
    d3_snr_d = my_snr[1]


    pk_rel_txt = pk_rel_txt + f'{d3_snr_d:.1f},{d3_snr_g:.1f}\n'
    critperc = 0.5
    print(f"{max_d:.1f},{max_g:.1f},{max_d3:.1f}")
    if (max_d> critperc*max_d3) & (max_g> critperc*max_d3):
    
    #if (np.nanmax(pkvals[ind_d1,ctr_d1_ind])> np.nanmax(pkvals[ind_d3,ctr_d3_ind])) & (np.nanmax(pkvals[ind_g,ctr_g_ind])> np.nanmax(pkvals[ind_d3,ctr_d3_ind])):
        mytxtcolor = 'k'
    else:
        mytxtcolor='r'
    #####
    #
    #   Handle q
    #         
    mycolor='C2'
    mydf_init,mydf_fit = ft.read_fit_result(path_q)
    
    mydf_fit_spc = mydf_fit[mydf_fit.spc_id == all_spc_ids[i]]

    yv = np.zeros(xv.shape)
    
    npks = len(mydf_init)
    pkvals = np.full((npks,len(xv)),np.nan)

    for index,mypeak in mydf_init.iterrows():
        pk_nm = mypeak.pk_name
        if (mypeak.pk_type == 'Lorentzian'):
            pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values.item()
            pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values.item()
            pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values.item()
            pk_m = ft.lorentzian(xv,pk_c,pk_s,pk_a)
            
        elif ((mypeak.pk_type == 'BWF_spc')):
            pk_c =  mydf_fit_spc[str(pk_nm+'_center')].values.item()
            pk_s =  mydf_fit_spc[str(pk_nm+'_sigma')].values.item()
            pk_a =  mydf_fit_spc[str(pk_nm+'_amplitude')].values.item()
            pk_q =  mydf_fit_spc[str(pk_nm+'_q')].values.item()
            pk_m = ft.BWF_spc(xv,pk_c,pk_s,pk_a,pk_q) 
            q_init_val = mydf_init.iloc[index]['pk_par4']
            #q_final_val[j,i] = mydf_fit_spc['g_q']

        yv = yv + pk_m    
        pkvals[index,:] = pk_m
        
        if  ('d1' in pk_nm):
            q_d1c = pk_c
            pkind_qd1 = rt.find_nearest(xv,q_d1c)
            sig_d1 = pk_s
        elif ('g' in pk_nm):
            q_base_val = pk_q
            q_gc = pk_c 
            pkind_qg = rt.find_nearest(xv,l2_gc) 
            sig_g = pk_s              
           

    #q_redchisq[j,i] = mydf_fit_spc.chisq_red.values
    print(f"q:{q_d1c:.1f},{q_gc:.1f}")
    q_redchisq_base = mydf_fit_spc.chisq_red.values.item() 
    noise = calcnoise(meas_data_shrt,yv)
    q_rmse = np.sqrt(mydf_fit_spc.mse.values.item())
    my_snr = calc_snr(pkvals,meas_data_shrt)
    q_snr_g = my_snr[0]
    q_snr_d = my_snr[1]
    
    mycolor='C2'
    mylinestyle = ':'

    h_lb, = myax.plot(xv,yv,color=mycolor,linewidth=2,linestyle=mylinestyle) 
    
    noise = calcnoise(meas_data_shrt,yv) 
    pk_rel_txt = pk_rel_txt + f'{q_snr_d:.1f},{q_snr_g:.1f}\n' 
    
    #####
    #
    #   End of section!!
    #
    file_id = all_spc_ids[i].replace("selected_","").replace(".txt","")
    figtxt = f"Figure 8{mylabel[i]}"
    #tabletxt = f"{spectrum_id[i]},{file_id}, {figtxt},{l2_redchisq:.1f} ,{d3_redchisq:.1f}, {q_redchisq_base:.1f}, {l2_d1c:.1f},{d3ctr_d1:.1f},{q_d1c:.1f},{l2_gc:.1f},{d3ctr_g},{d3ctr_d3},{q_gc:.1f},{q_base_val:.1f},{l2_snr_g:.1f},{d3_snr_g:.1f},{q_snr_g:.1f},{l2_snr_d:.1f},{d3_snr_d:.1f},{q_snr_d:.1f}\n"
    #tabletxt =            f"{i+1},{file_id}, {figtxt},{l2_redchisq:.1f} ,{d3_redchisq:.1f}, {q_redchisq_base:.1f}, {l2_d1c:.1f},{d3d1_base_c},{q_d1c:.1f},{l2_gc:.1f},{d3g_base_c},{d3d3_base_c},{q_gc:.1f}            ,{l2_snr_g:.1f},{d3_snr_g:.1f},{q_snr_g:.1f},{l2_snr_d:.1f},{d3_snr_d:.1f},{q_snr_d:.1f}\n"
    tabletxt = f"{spectrum_id[i]},{file_id},, {figtxt},{l2_rmse:.1f} ,{d3_rmse:.1f}, {q_rmse:.1f}, {l2_d1c:.1f},{d3d1_base_c},{q_d1c:.1f},{l2_gc:.1f},{d3g_base_c},{q_gc:.1f},{d3d3_base_c},{q_base_val:.1f},{l2_snr_g:.1f},{d3_snr_g:.1f},{q_snr_g:.1f},{l2_snr_d:.1f},{d3_snr_d:.1f},{q_snr_d:.1f}\n"
       
    #if i > 5:
    #Key point here: we write them all just to serve as a 'check'
    with open("dg_Table1_v1a-chk.csv",'a') as fh:    
        fh.write(tabletxt)  
    
    #Finish data!!
    myax.text(0.98,0.95,pk_rel_txt,transform=myax.transAxes,horizontalalignment='right',verticalalignment='top',color=mytxtcolor) #
    #myax.text(0.02,0.95,mylabel[tncntr],transform=myax.transAxes,verticalalignment='top',)
    myax.xaxis.set_tick_params(direction='in',which='both')
    
    #if ccntr == 1:
    #    myax.yaxis.set_ticks_position("right")
    
    ncntr=ncntr+1
    tncntr=tncntr+1
    
    if ncntr > plot_n_rows-1:      
        ccntr = ccntr+1
        ncntr=0
        
    mhl = 0.75
    myax.set_xticklabels("")
    if i == 0:
        myax.text(0.02,0.9,"A.",transform=myax.transAxes)

    if i == 4:
        myax.text(0.02,0.9,"B.",transform=myax.transAxes)   

    if i == 6:
        myax.text(0.02,0.9,"C.",transform=myax.transAxes)
  
    if i == 1:

        myax.legend([h_meas,h_l2,h_l2g,h_l2g_i,h_lb],['Meas','L2','L2G','L2Gpk','LB'],handlelength=mhl)
    if i == 5:

        myax.legend([h_meas,h_l2,h_l2g,h_l2g_i,h_lb],['Meas','L2','L2G','L2Gpk','LB'],handlelength=mhl)
    if i == 7:

        myax.legend([h_meas,h_l2,h_l2g,h_l2g_i,h_lb],['Meas','L2','L2G','L2Gpk','LB'],handlelength=mhl)

f.subplots_adjust(hspace=0.05,wspace=0.05)
ax[plot_n_rows-1,0].set_xticks([1000,1200,1400,1600,1800])    
ax[plot_n_rows-1,1].set_xticks([1000,1200,1400,1600,1800])
ax[plot_n_rows-1,0].set_xlabel(r'Wavenumber')
ax[plot_n_rows-1,1].set_xlabel(r'Wavenumber ')
ax[plot_n_rows-2,0].set_ylabel(r'                                          Raman Intensity')
ax[plot_n_rows-3,1].yaxis.set_label_position("right")
ax[plot_n_rows-3,1].set_ylabel(r'                                          Raman Intensity')
ax[plot_n_rows-1,1].yaxis.set_label_position("right")
ax[plot_n_rows-1,1].set_ylabel(r'                                          Raman Intensity')

#.canvas.draw()

#myax.legend([h_meas,h_l2,h_l2g,h_l2g_i,h_lb],['meas/lin','L2 Fit','L2G Fit','L2G Peaks','LB Fit'])
plt.savefig("Figure10.png")


