
#import matplotlib
#matplotlib.use('Agg') #Uncomment for production figure generation. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rs_tools as rt
rt.plot_format(10)

import os
import itertools
import io

def BWF_spc(x,center,sigma,amplitude,q):
    """ Return a Breit-Wigner-Fano lineshape
        The function here is used by the Raman Spectroscopy/Carbon community
        See Ferrari and Robertson (2000) for details.
        Note that here 'amplitude' is in terms of the peak height
    """
    gamma = sigma
    return amplitude*(1+2*(x-center)/(gamma*q))**2 / (1+(2*(x-center)/gamma)**2)


def gaussian(xv,x0,sigma,A):
    #This is the function as defined by lmfit
    #Added here in case you didn't import LMFIT

    gaus = (A/(sigma*np.sqrt(2*np.pi)))*np.exp(-1*np.divide((xv-x0)**2,2*sigma**2))
    #Something not right about the gausssian. 
    return gaus

def lorentzian(xv,x0,sigma,A):
    #This is the function as defined by lmfit
    #Added here in case you didn't import LMFIT
    lor = (A/np.pi)*np.divide(sigma,(xv-x0)**2+sigma**2)
    return lor

def breitwignerfano(xv,x0,sigma,A,q):
    #This is the function as defined by lmfit
    #Added here in case you didn't import LMFIT
    bwf = (A*((q*sigma/2)+xv-x0)**2)/((sigma/2)**2+(xv-x0)**2)#(A/np.pi)*np.divide(sigma,(xv-x0)**2+sigma**2)
    
    return bwf

def height_to_amplitude(height,sigma,q=0,gamma=0,curve='Lorentzian'):
    #Given a function height, converts that height to the 'amplitude' 
    #value. This is the value used by LMFit
    
    if curve== 'Lorentzian':
        amplitude = height*sigma*np.pi
    elif curve == 'Gaussian':
        amplitude = height*sigma*np.sqrt(2*np.pi)
    elif curve == 'BreitWignerFano':
        amplitude = height/(q**2)
    elif curve == 'BWF_spc':
        amplitude = height
    else:
        print("Invalid curve, defaulting to Lorentzian")
        amplitude = height*sigma*np.pi        
    return amplitude

def fwhm_to_sigma_gaussian(fwhm):
    #For a gaussian, given FWHM, return what LMFIT should use as sigma. 
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    return sigma

def fwhm_to_sigma_lorentzian(fwhm):
    # For a lorentzian with input fwhm return the siga
    sigma = fwhm/2.0
    return sigma


def read_fit_result(myfile,skipinirows=3):
    #Reader for the output files, reads into two data frames. 
    #Helped by: https://stackoverflow.com/questions/39724298/pandas-extract-comment-lines
    with open(myfile,'r') as f:
        header_itr = itertools.takewhile(lambda myline:myline.startswith('#'),f)
        header = list(header_itr)
    initialization_list = header[skipinirows::]
    initialization_list_clean = [myline[1::] for myline in initialization_list]
    
    mydf_init = pd.read_csv(io.StringIO('\n'.join(initialization_list_clean)),sep='\\s+')
    
    mydf_fit = pd.read_csv(myfile,comment='#',sep=',',skipfooter=1,engine='python')
    
    #From: https://stackoverflow.com/questions/42171709/creating-pandas-dataframe-from-a-list-of-strings   
    return mydf_init,mydf_fit



def calcnoise(measured,modeled):
    #meas_data_shrt-yv
    residual = measured-modeled
    resi_mean = np.nanmean(residual)
    residual_adj = residual-resi_mean
    noise = np.sqrt(np.nansum(residual_adj**2)/len(residual))
    return noise

           
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

                    myax_lp.plot([0,0],[-diag_len,0.90],transform=myax_lp.transAxes,color='k',clip_on=False)



            except Exception as err:
                import pdb
                pdb.set_trace()
            

def plot_init_final_fit(ax1,ax2,xv,my_fit_init,my_fit,cind='C0',csum='C1'):
    #ax1 = axtoc1[0,0]
    #ax2 = axtoc1[0,1]
    #my_fit_spc = mydf_fit_spc
    #my_df_init = mydf_init
    
    yv = np.zeros(xv.shape) 
    init_yv = np.zeros(xv.shape) 

    for index,mypeak in my_fit_init.iterrows():
        pk_nm = mypeak.pk_name
        pk_c =  my_fit[str(pk_nm+'_center')].values
        pk_s =  my_fit[str(pk_nm+'_sigma')].values
        pk_a =  my_fit[str(pk_nm+'_amplitude')].values
        
        init_pk_c =  my_fit[str('init_' +pk_nm+'_center')].values
        init_pk_s =  my_fit[str('init_' +pk_nm+'_sigma')].values
        init_pk_a =  my_fit[str('init_' +pk_nm+'_amplitude')].values            

        
        if ((mypeak.pk_type == 'Lorentzian')):
            pk_m = lorentzian(xv,pk_c,pk_s,pk_a)
            init_pk_m = lorentzian(xv,init_pk_c,init_pk_s,init_pk_a)
        elif ((mypeak.pk_type == 'BreitWignerFano')):
            pk_q =  my_fit[str(pk_nm+'_q')].values
            init_pk_q =  my_fit[str('init_' + pk_nm+'_q')].values
            pk_m = breitwignerfano(xv,pk_c,pk_s,pk_a,pk_q) 
            init_pk_m = breitwignerfano(xv,init_pk_c,init_pk_s,init_pk_a,init_pk_q)
        elif ((mypeak.pk_type == 'BWF_spc')):
            pk_q =  my_fit[str(pk_nm+'_q')].values
            
            init_pk_q =  my_fit[str('init_' + pk_nm+'_q')].values
            pk_m = BWF_spc(xv,pk_c,pk_s,pk_a,pk_q)
            init_pk_m = BWF_spc(xv,init_pk_c,init_pk_s,init_pk_a,init_pk_q)
            
            #import pdb
            #pdb.set_trace()
            
        elif (mypeak.pk_type == 'Gaussian'):
            pk_m = gaussian(xv,pk_c,pk_s,pk_a) 
            init_pk_m = gaussian(xv,init_pk_c,init_pk_s,init_pk_a)
            
        hi_ind, = ax1.plot(xv,init_pk_m,color=cind,linewidth=1,linestyle='--')                
        hf_ind, = ax2.plot(xv,pk_m,color=cind,     linewidth=1,linestyle='--')                

        yv = yv + pk_m
        init_yv = init_yv + init_pk_m
    hi_sum, = ax1.plot(xv,init_yv,color=csum)           
    hf_sum, = ax2.plot(xv,yv,     color=csum)
 
    return (hi_ind,hi_sum,hf_ind,hf_sum)          
         
        
spectrum_directory = "data_files"
#xlims=[800,2000]
xlims=[900,1800]
#Indices for the ALS fit
ind_s = 133
ind_e = 647

#TOC Plot 1


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

spclen = len(all_spc_ids)

pkH = 1
fwhm=100

pk_sig_l = fwhm/2
pk_sig_g = fwhm/2.3548


pkA_g = height_to_amplitude(pkH,pk_sig_g,curve='Gaussian')
pkA_l = height_to_amplitude(pkH,pk_sig_l,curve='Lorentzian')
#axtoc_leg = ftoc1.add_subplot(gs[0,4])


pk_c = 1600

pk_x = np.arange(1200,2000)

pk_g = gaussian(pk_x,pk_c,pk_sig_g,pkA_g)
pk_l = lorentzian(pk_x,pk_c,pk_sig_l,pkA_l)
pk_q = BWF_spc(pk_x,pk_c,fwhm,pkH,-10)
pk_q1 = BWF_spc(pk_x,pk_c,fwhm,pkH,-5)
pk_q2 = BWF_spc(pk_x,pk_c,fwhm,pkH,-1*10**5)

plt.figure()
plt.plot(pk_x,pk_g,color=[0.5,0.5,0.5])
plt.plot(pk_x,pk_l,color='k')
plt.plot(pk_x,pk_q1,color="C0")
plt.plot(pk_x,pk_q,color="C1")
plt.plot(pk_x,pk_q2,":C2")

plt.legend(["Gaussian","Lorentzian","BWF,Q=-5","BWF,Q=-10","BWF,Q=-10$^5$"])
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.yticks([])
plt.xlim([1200,2000])
plt.ylim(-0.01,pkH*1.1)
plt.savefig("FigS1.png")


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

xticks = [1050,1250,1450,1650]


d_final_val_all = np.full((spclen,len(d1_init_pos_data)),np.nan)
d_init_val_all = np.full((spclen,len(d1_init_pos_data)),np.nan)

g_final_val_all = np.full((spclen,len(d1_init_pos_data)),np.nan)
g_init_val_all = np.full((spclen,len(d1_init_pos_data)),np.nan)


##############################################################################
#
# 1 Lor 1 BWF
#
# q_load_arr = [     
#         '+q1000',
#         '+q100',
#         '+q50',
#         '+q10',
#         '+q5',
#         '+q4',
#         '+q3',
#         '+q2',        
#         '+q1',
#         '-q05',         
#         '-q1',
#         '-q1',
#         '-q2',        
#         '-q3',        
#         '-q4',        
#         '-q5',
#         '-q6',
#         '-q7',
#         '-q8',
#         '-q9',
#         '',
#         '-q11',
#         '-q12',
#         '-q13',
#         '-q14',
#         '-q15',
#         '-q16',        
#         '-q18',
#         '-q30',
#         '-q40',
#         '-q50',
#         '-q100',
#         '-q500',
#         '-q1000',
#         '-q10000']
#
#q_base = 16

q_load_arr = [     
        '-q3',               
        '-q5',
        '-q9',
        '-q30',
        '-q10000']

q_basea = 1
q_baseb = 4
q_crit=3
q_text_base = 'dg_2pk_bwflor-apprx'


qplottxt = "FigS5_alldata.png"
qplottxtz = "FigS6_alldata_zoom.png"
q_dir = 'fit_results'
#


##############################################################################
#
# 2 Lor 1 Gaus
#
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


###############################################################################
#
#
#



pk5_load_arr = [
        '-var3',
        '-var2',
        '-noc',
        '',]

run_desc = "a:LoGHiD2;b:SDZ633;c:COM-NC;d:COM"
colors_5pk = ['C3','C2','C0','C1']
#j_name = ['Com','ComNC','HiD2LoG','SDZ633']


pk5_text_base = 'dg_5pk_c_smp-ratapprx'

input_dir = l2_dir

d1_c = np.full((len(pk5_load_arr),spclen),np.nan)
d1_s = np.full((len(pk5_load_arr),spclen),np.nan)
d1_a = np.full((len(pk5_load_arr),spclen),np.nan)
g_c = np.full((len(pk5_load_arr),spclen),np.nan)
g_s = np.full((len(pk5_load_arr),spclen),np.nan)
g_a = np.full((len(pk5_load_arr),spclen),np.nan)
g_h = np.full((len(pk5_load_arr),spclen),np.nan)
d2_c = np.full((len(pk5_load_arr),spclen),np.nan)
d2_s = np.full((len(pk5_load_arr),spclen),np.nan)
d2_a = np.full((len(pk5_load_arr),spclen),np.nan)
d2_h = np.full((len(pk5_load_arr),spclen),np.nan)

fign=1

figlab_a = ["A.","B.","C.","D.","E.","F.","G.","H."]
figlab_b = ["A.","B.","C.","D.","E.","F.","G.","H."]
#figlab_b = ["I.","J.","K.","L.","M.","N.","O.","P."]

mfs=10
mfs1=10
mfs2=8
mfw = 'bold'
#Create dummy axis handle
#f,dax1 = plt.subplots(4,2)
#f,dax2 = plt.subplots(4,2)

for i in range(spclen):
    
    #if i == 9:
    #    pptxt = 'Fig_TOC'
    #else:
    #    pptxt = 'Extra_TOC'>
    
    status='A'
    pptxt = f"Fig_SPC{i+1}"
    myfiglab=figlab_a

    
    # DD 3/12
    ftoc1,axtoc1 = plt.subplots(nrows=4,ncols=2,sharex=True,sharey=True,figsize=[6,12],dpi=300)
    #ftoc1,axtoc1 = plt.subplots(nrows=4,ncols=2,sharex=True,sharey=True,figsize=[12,6],dpi=300)
    #ftoc1.suptitle(f'Figure {i+1} Spectrum {i+1}', fontsize=mfs,fontweight=mfw)

    
    #####
    #
    #   Handle 2 lor
    #

    myinputfile = all_spc_ids[i]
    #output_figure_name = pptxt + '_v1_' + myinputfile.replace('.txt','').replace('selected_','') + '.png'
    output_figure_name = pptxt + '_' + myinputfile.replace('.txt','').replace('selected_','') + '_nounc.png'

    print(myinputfile)
    mydata = np.loadtxt(os.path.join(spectrum_directory,myinputfile),delimiter=' ',skiprows=0)
    data_x = mydata[:,0]
    data_y = mydata[:,1]
    x_min_ind = rt.find_nearest(data_x,xlims[0])
    x_max_ind = rt.find_nearest(data_x,xlims[1])+1      
    xv = data_x[x_min_ind:x_max_ind]
    
  
    
    #loop through all peak position values 
    for j in range(len(l2_load_arr)):
        myfitfile = l2_text_base + l2_load_arr[j] + '.csv'
        myfitpath= os.path.join(l2_dir,myfitfile)
        mydf_init,mydf_fit = read_fit_result(myfitpath)        
        mydf_fit_spc = mydf_fit[mydf_fit.spc_id == myinputfile]

        
        if j == 0:
            meas_spc_sub=data_y-data_x*mydf_fit_spc.bkr_s.values-mydf_fit_spc.bkr_c.values
            maxv = np.nanmax(meas_spc_sub[ind_s:ind_e])      
            meas_data_shrt = meas_spc_sub[x_min_ind:x_max_ind]
            maxy = np.nanmax(meas_data_shrt)
            hm1, = axtoc1[0,0].plot(data_x,meas_spc_sub,color='k',alpha=0.8)
            hm2, =axtoc1[0,1].plot(data_x,meas_spc_sub,color='k',alpha=0.8)
            hm3, =axtoc1[3,0].plot(data_x,meas_spc_sub,color='k',alpha=0.8)
            hm4, =axtoc1[3,1].plot(data_x,meas_spc_sub,color='k',alpha=0.8) 
            hm5, =axtoc1[1,0].plot(data_x,meas_spc_sub,color='k',alpha=0.8)
            hm6, =axtoc1[1,1].plot(data_x,meas_spc_sub,color='k',alpha=0.8)
            hm7, =axtoc1[2,0].plot(data_x,meas_spc_sub,color='k',alpha=0.8)
            hm8, =axtoc1[2,1].plot(data_x,meas_spc_sub,color='k',alpha=0.8)

            fine_format_axis(axtoc1)




        ax1 = axtoc1[0,0]
        ax2 = axtoc1[0,1]
        
        mycind = 'C9'
        mycall =  'C1'  
        
        handles = plot_init_final_fit(axtoc1[0,0],axtoc1[0,1],xv,mydf_init,mydf_fit_spc,cind='C9',csum='C1')
        if j == l2_base:
            h1_ind =  handles[0]
            h1_sum =  handles[1]
            h2_ind =  handles[2]
            h2_sum =  handles[3]
            

    #1L1BWF
   #####
    #
    #   Handle q
    #         

        
    for j in range(len(q_load_arr)):
        yv = np.zeros(xv.shape) 
        myfitfile = q_text_base + q_load_arr[j] + '.csv'
        myfitpath= os.path.join(q_dir,myfitfile)
        mydf_init,mydf_fit = read_fit_result(myfitpath)
        
        mydf_fit_spc = mydf_fit[mydf_fit.spc_id == all_spc_ids[i]]
        
        if j > q_crit:
            mycind = 'C9'
            mycall =  'C1'
        else:
            mycind = 'C6'
            mycall = 'C2'
        
        handles = plot_init_final_fit(axtoc1[1,0],axtoc1[1,1],xv,mydf_init,mydf_fit_spc,cind=mycind,csum=mycall)
        
        if j == q_basea:
            hqifa = handles[0]
            hqisa = handles[1]
            hqffa = handles[2]
            h1fsa = handles[3]
        if j == q_baseb:
            hqifb = handles[0]
            hqisb = handles[1]
            hqffb = handles[2]
            h1fsb = handles[3] 
         
        
    #3 pK

    csum='C1'
    cind='C9'

    for j in range(len(d3_load_arr)):
        yv = np.zeros(xv.shape) 
            
        myfitfile = d3_text_base + d3_load_arr[j] + '.csv'
        myfitpath= os.path.join(d3_dir,myfitfile)
        
        mydf_init,mydf_fit = read_fit_result(myfitpath)
        
        mydf_fit_spc = mydf_fit[mydf_fit.spc_id == all_spc_ids[i]]
        
        handles = plot_init_final_fit(axtoc1[2,0],axtoc1[2,1],xv,mydf_init,mydf_fit_spc,cind=mycind,csum=mycall)
        
        if j == d3_base:
            h3ifa = handles[0]#h_jnki_1
            h3isa = handles[1]#h_jnki_2
            h3ffa = handles[2]#h_jnkf_1
            h3fsa = handles[3]#h_jnkf_2
            
            
    #5pk
    jcols = colors_5pk
    print("5pk")  
    #loop through all 5 peak values
    for j in range(len(pk5_load_arr)):
        print(j,pk5_load_arr[j])

        
        mycall=jcols[j]
        mycind=jcols[j]
        
        myfitfile = pk5_text_base + pk5_load_arr[j] + '.csv'
        mydf_init,mydf_fit = read_fit_result(os.path.join(input_dir,myfitfile))
        
        full_spc_ids = clean_spc_ids(mydf_fit.spc_id.copy())        
        
        mydf_fit_spc = mydf_fit[full_spc_ids == myinputfile]
        
        handles = plot_init_final_fit(axtoc1[3,0],axtoc1[3,1],xv,mydf_init,mydf_fit_spc,cind=mycind,csum=mycall)

        if j == 0:
            h1_inda = handles[0]#h_jnki_3
            h1_suma = handles[1]#h_jnki_4  
            h2_inda = handles[2]#h_jnkf_3
            h2_suma = handles[3]#h_jnkf_4           
        elif j == 1:
            h1_indb = handles[0]#h_jnki_3
            h1_sumb = handles[1]#h_jnki_4 
            h2_indb = handles[2]#h_jnkf_3
            h2_sumb = handles[3]#h_jnkf_4   
        elif j == 2:
            h1_indc = handles[0]#h_jnki_3
            h1_sumc = handles[1]#h_jnki_4 
            h2_indc = handles[2]#h_jnkf_3
            h2_sumc = handles[3]#h_jnkf_4 
        elif j == 3:
            h1_indd = handles[0]#h_jnki_3
            h1_sumd = handles[1]#h_jnki_4 
            h2_indd = handles[2]#h_jnkf_3
            h2_sumd = handles[3]# h_jnkf_4 
        
    axtoc1[0,0].set_xlim([1000,1700]) 

    #Here handle formatting that is the same for every row
    for j in range(4):

        axtoc1[j,0].set_ylabel(r"Raman Intensity",fontsize=mfs2)
        axtoc1[j,1].set_xlabel(r"Wavenumber \ cm$^{-1}$",fontsize=mfs2)
        axtoc1[j,0].set_xlabel(r"Wavenumber \ cm$^{-1}$",fontsize=mfs2)
        axtoc1[j,0].text(0.5,0.92,"Starting Functions (LS)",transform=axtoc1[j,0].transAxes,horizontalalignment='center',verticalalignment='top',fontsize=mfs2)
        axtoc1[j,1].text(0.5,0.92,"Optimized Functions (LS)",transform=axtoc1[j,1].transAxes,horizontalalignment='center',verticalalignment='top',fontsize=mfs2)

    #axtoc1[0,0].set_title("Starting Functions (ST)",fontsize=mfs1,fontweight=mfw)
    #axtoc1[0,1].set_title("Optimized Functions (OPT)",fontsize=mfs1,fontweight=mfw)
    #figlab_a = ["A.","B.","C.","D.","E.","F.","G.","H."]
    #axtoc1[0,0].legend([hm1,h1_sum,h1_ind],['Measured',r'$\Sigma$ST','ST '],loc='upper left',bbox_to_anchor=(0.01, 0.91),fontsize=mfs2,handlelength=1,frameon=False,ncol=1,labelspacing=0.2,columnspacing=1)
    axtoc1[0,1].legend([hm2,h2_sum,h2_ind],['Measured',r'$\Sigma$Lineshapes','  Lineshapes'],loc='upper left',bbox_to_anchor=(-0.1, 0.89),fontsize=mfs2,handlelength=1,frameon=False,ncol=1,labelspacing=0.2,columnspacing=1)
    axtoc1[0,0].set_xticks(xticks)
    axtoc1[0,0].text(-0.1,0.98,f"{myfiglab[0]} Two Lorentzian",transform=axtoc1[0,0].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=mfs1)
    #axtoc1[0,1].text(0.05,0.98,f" OPT 2 Lorentzian, 9 init sets",transform=axtoc1[0,1].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=8)
    ftoc1.suptitle(f"Spectrum {i+1}",fontsize=mfs1,fontweight='bold')
    #axtoc1[1,0].legend([hm5,hqisa,hqifa,hqisb,hqifb],['Measured',r'$\Sigma$ST_a','ST_a',r'$\Sigma$ST_b','ST_b'],bbox_to_anchor=(0.01, 0.91),loc='upper left',fontsize=8,handlelength=2,frameon=False,ncol=1,columnspacing=1)
    axtoc1[1,1].legend([hm6,h1fsa,hqffa,h1fsb,hqffb],['Measured',r'$\Sigma$LS Q>-10$^4$',r'  LS Q>-10$^4$',r'$\Sigma$LS Q=-10$^4$',r'  LS Q=-10$^4$',],bbox_to_anchor=(-0.1, 0.89),loc='upper left',fontsize=8,handlelength=2,frameon=False,ncol=1,columnspacing=1)
    axtoc1[1,0].set_xticks(xticks)
    #axtoc1[1,0].text(0.05,0.98,f"{myfiglab[1]}"+" ST 1 Lorentzian/1 BWF\n a:Q$_{init}$=-3,-6,-9,-30;  b:Q$_{init}$=-10$^4$",transform=axtoc1[1,0].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=8)
    axtoc1[1,0].text(-0.1,0.98,f"{myfiglab[1]} One Lorentzian/One BWF" ,transform=axtoc1[1,0].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=mfs1)
    #axtoc1[1,0].text(0.5,0.93,"Starting Functions (ST)",transform=axtoc1[1,0].transAxes,horizontalalignment='center',verticalalignment='top',fontsize=mfs2)
    #axtoc1[1,1].text(0.5,0.93,"Optimized Functions (OPT)",transform=axtoc1[1,1].transAxes,horizontalalignment='center',verticalalignment='top',fontsize=mfs2)
    #axtoc1[1,1].text(0.05,0.98,f" "+"OPT Lorentzian/1 BWF\n a:Q$_{init}$=-3,-6,-9,-30;  b:Q$_{init}$=-10$^4$",transform=axtoc1[1,1].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=mfs1)    

    #axtoc1[2,0].legend([hm7,h3isa,h3ifa],['Measured',r'$\Sigma$ST','ST',r'$\Sigma$ST_b','ST_b'],bbox_to_anchor=(0.01, 0.91),loc='upper left',fontsize=8,handlelength=2,frameon=False,ncol=1,columnspacing=1)
    axtoc1[2,1].legend([hm8,h3fsa,h3ffa],['Measured',r'$\Sigma$Lineshape','  Lineshape',r'$\Sigma$OPT_b','OPT_b'],bbox_to_anchor=(-0.1, 0.89),loc='upper left',fontsize=8,handlelength=2,frameon=False,ncol=1,columnspacing=1)
    axtoc1[2,0].set_xticks(xticks)
    #axtoc1[2,0].text(0.05,0.98,f"{myfiglab[2]}  ST 2 Lorentzian/1 Gaussian",transform=axtoc1[2,0].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=mfs1)
    axtoc1[2,0].text(-0.1,0.98,f"{myfiglab[2]} Two Lorentzian/One Gaussian",transform=axtoc1[2,0].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=mfs1)
    #axtoc1[2,1].text(0.05,0.98,f"OPT 2 Lorentzian/1 Gaussian",transform=axtoc1[2,1].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=8)    

    #axtoc1[3,0].legend([hm3,h1_suma,h1_inda,h1_sumb,h1_indb,h1_sumc,h1_indc,h1_sumd,h1_indd],['Measured',r'$\Sigma$STa','STa',r'$\Sigma$STb','STb',r'$\Sigma$STc','STc',r'$\Sigma$STd','STd'],bbox_to_anchor=(0.01, 0.91),loc='upper left',fontsize=8,handlelength=2,frameon=False,ncol=1,columnspacing=1)
    axtoc1[3,1].legend([hm4,h2_suma,h2_inda,h2_sumb,h2_indb,h2_sumc,h2_indc,h2_sumd,h2_indd],['Measured',r'$\Sigma$LS LoGHiD2','  LS LoGHiD2',r'$\Sigma$LS SDZ633','  LS SDZ633',r'$\Sigma$LS COM-NC','  LS COM-NC',r'$\Sigma$LS COM','  LS COM'],bbox_to_anchor=(-0.1, 0.89),loc='upper left',fontsize=8,handlelength=2,frameon=False,ncol=1,columnspacing=1)
    axtoc1[3,0].set_xticks(xticks)
    #axtoc1[3,0].text(0.02,0.98,f"{myfiglab[6]} ST 4 Lorentzian/1 Gaussian\n" + run_desc,transform=axtoc1[3,0].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=mfs1)
    axtoc1[3,0].text(-0.1,0.98,f"{myfiglab[3]} Four Lorentzian/One Gaussian",transform=axtoc1[3,0].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=mfs1)
    #axtoc1[3,1].text(0.02,0.98,f" OPT 4 Lorentzian/1 Gaussian\n"+run_desc,transform=axtoc1[3,1].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=8)    
    


    axtoc1[0,0].set_ylim([0,1.3*maxy])
     
    #axtoc1[3,1].set_xlabel(r"Raman Shift \ cm$^{-1}$")
    #axtoc1[3,0].set_xlabel(r"Raman Shift \ cm$^{-1}$")
    #axtoc1[0,0].set_ylabel(r"Raman Intensity")
    #axtoc1[1,0].set_ylabel(r"Raman Intensity")
    #axtoc1[2,0].set_ylabel(r"Raman Intensity")
    #axtoc1[3,0].set_ylabel(r"Raman Intensity")
    axtoc1[0,0].set_yticks([])
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.02,wspace=0.03)

    ftoc1.savefig(output_figure_name)
    #if i == 11:
    #    please_stop_here()
    
    plt.close()
    
 
 
 
 #####
 #
 #   TOC
 #
print("Generating TOC Figure--------------------") 
 
ftoc1,axtoc1 = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize=[6,6])

 

myinputfile = all_spc_ids[2]
#output_figure_name = pptxt + '_v1_' + myinputfile.replace('.txt','').replace('selected_','') + '.png'
output_figure_name = "TOC.png"
print(myinputfile)
mydata = np.loadtxt(os.path.join(spectrum_directory,myinputfile),delimiter=' ',skiprows=0)
data_x = mydata[:,0]
data_y = mydata[:,1]
x_min_ind = rt.find_nearest(data_x,xlims[0])
x_max_ind = rt.find_nearest(data_x,xlims[1])+1      
xv = data_x[x_min_ind:x_max_ind]
 
  
 
#loop through all peak position values 
for j in range(len(l2_load_arr)):
    myfitfile = l2_text_base + l2_load_arr[j] + '.csv'
    myfitpath= os.path.join(l2_dir,myfitfile)
    mydf_init,mydf_fit = read_fit_result(myfitpath)        
    mydf_fit_spc = mydf_fit[mydf_fit.spc_id == myinputfile]

 
    if j == 0:
        meas_spc_sub=data_y-data_x*mydf_fit_spc.bkr_s.values-mydf_fit_spc.bkr_c.values
        maxv = np.nanmax(meas_spc_sub[ind_s:ind_e])      
        meas_data_shrt = meas_spc_sub[x_min_ind:x_max_ind]
        maxy = np.nanmax(meas_data_shrt)
        hm1, = axtoc1[0,0].plot(data_x,meas_spc_sub,color='k',alpha=0.8)
        hm2, =axtoc1[0,1].plot(data_x,meas_spc_sub,color='k',alpha=0.8) 
        hm5, =axtoc1[1,0].plot(data_x,meas_spc_sub,color='k',alpha=0.8)
        hm6, =axtoc1[1,1].plot(data_x,meas_spc_sub,color='k',alpha=0.8)
    
    
    ax1 = axtoc1[0,0]
    ax2 = axtoc1[0,1]
     
    mycind ='C9'
    mycall ='C1'  
     
    handles = plot_init_final_fit(axtoc1[0,0],axtoc1[0,1],xv,mydf_init,mydf_fit_spc,cind='C9',csum='C1')
    if j == l2_base:
        h1_ind =  handles[0]
        h1_sum =  handles[1]
        h2_ind =  handles[2]
        h2_sum =  handles[3]
     

       
#5pk
jcols = colors_5pk
print("5pk")  

colors_5pk = ['C2','C1']

pk5_load_arr = [
        '-var2',
        '']

run_desc = "a & b have different init. conds."

#loop through all 5 peak values
for j in range(len(pk5_load_arr)):
    print(j,pk5_load_arr[j])
 
    mycall=jcols[j]
    mycind=jcols[j]
    
    myfitfile = pk5_text_base + pk5_load_arr[j] + '.csv'
    mydf_init,mydf_fit = read_fit_result(os.path.join(input_dir,myfitfile))
 
    full_spc_ids = clean_spc_ids(mydf_fit.spc_id.copy())        
 
    mydf_fit_spc = mydf_fit[full_spc_ids == myinputfile]
 
    handles = plot_init_final_fit(axtoc1[1,0],axtoc1[1,1],xv,mydf_init,mydf_fit_spc,cind=mycind,csum=mycall)

    if j == 0:
        h1_inda = handles[0]#h_jnki_3
        h1_suma = handles[1]#h_jnki_4  
        h2_inda = handles[2]#h_jnkf_3
        h2_suma = handles[3]#h_jnkf_4           
    elif j == 1:
        h1_indb = handles[0]#h_jnki_3
        h1_sumb = handles[1]#h_jnki_4 
        h2_indb = handles[2]#h_jnkf_3
        h2_sumb = handles[3]#h_jnkf_4   

axtoc1[0,0].set_xlim([1000,1700]) 

axtoc1[0,0].set_title("Starting Functions (ST)")
axtoc1[0,1].set_title("Optimized Functions (OPT)")

axtoc1[0,0].legend([hm1,h1_sum,h1_ind],['Measured',r'$\Sigma$ST','ST '],loc='upper left',bbox_to_anchor=(0.01, 0.965),fontsize=8,handlelength=1,frameon=False,ncol=3,labelspacing=0.2,columnspacing=1)
axtoc1[0,1].legend([hm2,h2_sum,h2_ind],['Measured',r'$\Sigma$OPT','OPT'],loc='upper left',bbox_to_anchor=(0.01, 0.965),fontsize=8,handlelength=1,frameon=False,ncol=3,labelspacing=0.2,columnspacing=1)
axtoc1[0,0].set_xticks(xticks)
axtoc1[0,0].text(0.05,0.98,"ST 2 Lorentzian, 9 init sets",transform=axtoc1[0,0].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=8)
axtoc1[0,1].text(0.05,0.98,"OPT 2 Lorentzian, 9 init sets",transform=axtoc1[0,1].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=8)

axtoc1[1,0].legend([hm3,h1_suma,h1_inda,h1_sumb,h1_indb],['Measured',r'$\Sigma$STa','STa',r'$\Sigma$STb','STb'],bbox_to_anchor=(0.01, 0.91),loc='upper left',fontsize=8,handlelength=2,frameon=False,ncol=1,columnspacing=1)
axtoc1[1,1].legend([hm4,h2_suma,h2_inda,h2_sumb,h2_indb],['Measured',r'$\Sigma$OPTa','OPTa',r'$\Sigma$OPTb','OPTb'],bbox_to_anchor=(0.01, 0.91),loc='upper left',fontsize=8,handlelength=2,frameon=False,ncol=1,columnspacing=1)
axtoc1[1,0].set_xticks(xticks)
axtoc1[1,0].text(0.05,0.98,"ST 4 Lorentzian/1 Gaussian\n" + run_desc,transform=axtoc1[1,0].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=8)
axtoc1[1,1].text(0.05,0.98,"OPT 4 Lorentzian/1 Gaussian\n"+ run_desc,transform=axtoc1[1,1].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=8)    
 
axtoc1[0,1].text(0.05,0.8,"All 9 starting \nfunctions \nconverge to \nsame \noptimized\nfunctions",transform=axtoc1[0,1].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=8)
axtoc1[1,1].text(0.6,0.87,"Optimized\nfunctions are\ndifferent for\nSTa and STb",transform=axtoc1[1,1].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=8)            


axtoc1[0,0].set_ylim([0,1.3*maxy])
axtoc1[1,0].set_xlabel(r"Wavenumber \ cm$^{-1}$")
axtoc1[1,1].set_xlabel(r"Wavenumber \ cm$^{-1}$")
axtoc1[0,0].set_ylabel(r"Raman Intensity")
axtoc1[1,0].set_ylabel(r"Raman Intensity")
axtoc1[0,0].set_yticks([])
 
plt.tight_layout()
plt.subplots_adjust(hspace=0.03,wspace=0.05)
ftoc1.savefig("TOC.png")
 
plt.close()
 
