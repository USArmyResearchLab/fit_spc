# -*- coding: utf-8 -*-
"""


"""
import multiprocessing
import time
import numpy as np
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import lmfit
from lmfit import models, model,Model
import pdb
import importlib
import argparse
from copy import deepcopy
from scipy.signal import savgol_filter
import logging
from pathlib import Path
import re


def eval_bool(input_str):
    #Takes a string and evaluates it to boolean True or False
    #This is safer than something like changing the type. 
    #However, you have to spell it correctly. 
    
    if  input_str.upper() == 'FALSE':
        result = False
    elif input_str.upper() == 'TRUE':
        result = True
    else:
        raise ValueError("Invalid Input type to eval_bool()")
    return result

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


def BWF_spc(x,center,sigma,amplitude,q):
    """ Return a Breit-Wigner-Fano lineshape
        The function here is used by the Raman Spectroscopy/Carbon community
        See Ferrari and Robertson (2000) for details.

        Note that here 'amplitude' is in terms of the peak height


    """
    gamma = sigma
    return amplitude*(1+2*(x-center)/(gamma*q))**2 / (1+(2*(x-center)/gamma)**2)


def read_fit_result(myfile,skipinirows=3):
    import itertools
    import io
    #Reader for the output files, reads into two data frames. 
    with open(myfile,'r') as fid:
        header_itr = itertools.takewhile(lambda myline:myline.startswith('#'),fid)
        header = list(header_itr)
    initialization_list = header[skipinirows::]
    initialization_list_clean = [myline[1::] for myline in initialization_list]
    
    mydf_init = pd.read_csv(io.StringIO('\n'.join(initialization_list_clean)),sep='\\s+')
    
    mydf_fit = pd.read_csv(myfile,comment='#',sep=',',skipfooter=1,engine='python')

    return mydf_init,mydf_fit


def fwhm_to_sigma_gaussian(fwhm):
    #For a gaussian, given FWHM, return what LMFIT should use as sigma. 
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    return sigma

def fwhm_to_sigma_lorentzian(fwhm):
    # For a lorentzian with input fwhm return the siga
    sigma = fwhm/2.0
    return sigma

def process_rs(rs):
    #This would be used for doing any preprocessing to the rs
    #Data saved as PNG appear to be 0-1. We convert back to ADC values (65535).
    adc_conv = 65535
    rs = rs*adc_conv
    return rs

def convert_rs_weight(rs):
    #Used if you want to calculate error on a RS. This is used to weight the fit model if 'weights' is set to 'true'. 
    #Here, we take in the "Raw" raman spectrum, and compute the error as sqrt(I/3) where I is the raman intensity. 
    # Note that we return the weight:1/w
    nreps=3
    err = np.sqrt(np.true_divide(rs,nreps))
    wght = np.true_divide(1,err)
    return wght


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


def amplitude_to_height(amplitude,sigma,q=0,gamma=0,curve='Lorentzian'):
    if curve == 'Lorentzian':
        height= amplitude*0.3183/sigma
    elif curve == 'Gaussian':
        height= amplitude*0.3989/sigma
    else:
        print("height calc not added yet....defaulting to gaussian")
        height=np.nan
    return height

def isValid(value):
    #Simplifies the calling of this function
    return pd.notna(value)
    

def handle_lum(input_x,input_y,ind_min,ind_max,remove_lum="None",remove_lum_delta=20):  
    ################
    #
    #   Added for lum remove
    #    
    #   baseline_const,baseline_slope,y = handle_lum(orig_x,orig_y,ind_min,ind_max,method=my_fit_options.remove_lum, rld= my_fit_options.remove_lum_delta) 
    #
    #   Currently only does linear fit. 
    #     

    x=input_x[ind_min:ind_max]
        
    if remove_lum == 'Linear':

        fit_ind_delta = remove_lum_delta
        fit_lower_max = ind_min+fit_ind_delta
        fit_upper_min = ind_max-fit_ind_delta
        fit_inds = np.r_[ind_min:fit_lower_max,fit_upper_min:ind_max]
        fit_x = input_x[fit_inds]
        fit_y = input_y[fit_inds]
        
        baseline_fit = np.polynomial.polynomial.polyfit(fit_x,fit_y,1)
        baseline_slope = baseline_fit[1]
        baseline_const_calc = baseline_fit[0]
        
        #y_prm=input_y[ind_min:ind_max,1]-(baseline_const_calc+baseline_slope*x)
        y_prm=input_y[ind_min:ind_max]-(baseline_const_calc+baseline_slope*x)

        #here change the constant so that the minimum in is zero (i.e. the function minimum cannot be <0
        #negar = np.nanmin([np.nanmedian(y_prm[ind_min:fit_lower_max]),np.nanmedian(y_prm[fit_upper_min:ind_max])])
        neg1 = np.nanmedian(y_prm[0:fit_ind_delta])
        neg2 = np.nanmedian(y_prm[-1*fit_ind_delta:-1])
        negar = np.nanmin([neg1,neg2])
        print(y_prm[fit_upper_min:ind_max-1],fit_upper_min,ind_max,len(y_prm))    
    

        baseline_const = baseline_const_calc+negar
        y=input_y[ind_min:ind_max]-(baseline_const+baseline_slope*x)
        print(f"Shifted:{baseline_const_calc:.4e},{baseline_const:.4e},{negar:.4e}")      
        
    else:
        baseline_const = np.nan
        baseline_slope = np.nan

        y=input_y[ind_min:ind_max] 

    return y,baseline_const,baseline_slope


def handle_wgt(weight_data,myind_min,myind_max,ncols,use_weights):
    #
    #   This function makes sure weights are correctly handled in the code
    #   Tries to deal with user error and avoid using the wrong value for the weights.
    #   
    #   It also sets the weight values for the fit, and changes the set_weights 
    #   variable so the code knows that weights have been set
    #
    
    set_weights=False
    if ncols==3 & use_weights==True:
        wgt=weight_data[myind_min:myind_max]
        set_weights=True
    elif ncols>3 & use_weights==True:
        wgt=weight_data[myind_min:myind_max]
        note = f"More columns:{ncols} provided....make sure weights is the third column"
        logging.warning(f"More columns:{ncols} provided....make sure weights is the third column")
        set_weights=True
    elif ncols==2 & use_weights==True:
        wgt=None
        note = "Asked for weights but no weights provided.....ignoring"
        logging.warning("Asked for weights but no weights provided.....ignoring")
        set_weights=False
    else:
        wgt=None
        note=""
        set_weights=False
        
    return wgt,note,set_weights

def get_init_intensity(my_x,my_y,my_my_model_parameters,amplitude_mode,print_result=True):
    #
    #   We need to provide the fitting code with an initial intensity/amplitude
    #   Generally that is going to be dependent on the data to be modeled.
    #   In these cases a savitzky-golay filter is used to smooth the data
    #   Options:
    #
    #       apprx: for each peak, take the function value at the inputted peak center. Use this one. 
    #       Other methods not reccomended, but left in for testing purposes:
    #       ratmax: not reccomended: take the maximum value of the function, and use some ratio related to that. 
    #       ratpos: not reccomended: take the intensity relative to the height of the first peak. You might do this if you wanted to test assumptions about relative peak heights       
    #       fixed: Just pick a fixed value
    #

    y_flt = savgol_filter(my_y,window_length=15,polyorder=2)    
    if 'ratmax' in amplitude_mode: #Reccomend to not use this one!
        my_use_intensity = np.nanmax(my_y)
    elif 'ratpos' in amplitude_mode:
        x_center = my_my_model_parameters.iloc[0].pk_center
        xdiff = np.abs(my_x-x_center)
        my_use_intensity = y_flt[xdiff == np.nanmin(xdiff)][0]
    elif 'apprx' in amplitude_mode:
        #Here we get the height at the initialization peak locations
        my_use_intensity = np.full(len(my_my_model_parameters),np.nan)
        for i in range(len(my_my_model_parameters)):
            x_center = my_my_model_parameters.iloc[i].pk_center
            xdiff = np.abs(my_x-x_center)
            my_use_intensity[i] = y_flt[xdiff == np.nanmin(xdiff)].item()
            
            #pdb.set_trace()
    elif amplitude_mode == 'fixed':
        my_use_intensity=1000
        logging.debug(f"default fixed intensity {my_use_intensity} selected")
    else:
        print("Incorrect specification of intensities")        
        logging.error("Incorrect specification of intensities")
        
    if print_result == True:
        print(amplitude_mode,my_use_intensity)
    logging.debug(str(my_use_intensity))
    return my_use_intensity

class fit_options_clean():
    
    _slots__ = ("fit_name","fig_output_dir","fit_notes","multicore","n_cores","n_single","spc_inds","file_rmta","file_rdat","file_rdat_raw","file_base","file_type","method")
    def __init__(self):
        self.fit_name = "default"
        self.log_datetime_str=""
        self.fit_options_name = ""
        self.fid_path = False #If the file ID is a path, include the path in the FID output array!! NOT IMPLEMENTED IN HANDLER YET!!
        self.log_name_str =""
        self.log_dir = "logs"
        self.fig_output_dir = "default"
        self.output_type='txtjson' #Options: txt, json, txtjson -> json saves every model output to a .json file
        
        self.fit_notes = "5 peak fit following Ivleva (4 lorentzians and one gaussian) "
        
        self.method='leastsq'
        
        self.multicore = False#Serial or parallel processing? 
        self.n_cores = 40 #How many cores do you want to use.
        self.output_resolution = '%.4e'    
        
        self.amplitude_mode = 'fixed'
        
        self.spc_inds= [242,511]
        self.remove_lum_delta=10 
        self.remove_lum='Linear'
        self.remove_lum_inds = np.r_[242:253,505:535] #Indices to remove luminescence
        ############
        #   Allowable spectrum types
        #   batch: batch process the files, search for all files in target_directory
        #   txt: batch process the files, search for all files in target_directory
        #   rdat: search through a .r mta and .rdat file
        #   
        self.spc_type="batch" #methods: 'batch', loop through all files matching files in directory
        
        self.target_directory=""
        

        
        #Options needed if method == batch
        self.file_base = "selected_"
        self.file_type = ".txt"
        
        #Options needed if method == batchtxt or method == txt
        #Currently, assumes the format is: First row/column: wavenumber, second row/column: intensity, third row/column(optional): raw intensity - used to compute accurate chi square
        
        self.comments = "#"
        self.delimiter=None
        self.skiprows=0
        self.usecols=None
        #self.use_weights=True
        self.use_weights=False
        
        self.fit_damping_parameter=100
    
        self.generate_figure=True
        
        #options needed if spctype = "rdat"
        self.file_rmta = "shortdata.rmta"
        self.file_rdat = "shortdata.rdat"
        self.file_rdat_raw = "shortdata_raw.rdat"
 
        self.n_single=71 #  Used to only select a single spectrum from a rdat. 
        


def build_model(my_fit_options,my_model_parameters,use_intensity=1000,output_peak_names=False):
    #
    #   Creates a LMFIT model 561
    #
    #
    #
    
    #use_intensity: this changes the intensity
    #import pdb
    #pdb.set_trace() 
    
    myq = 1 #Only modified if BWF
    print("a") 
    for index,peak in my_model_parameters.iterrows():
    #peak_name	peak_type	peak_center	peak_sigma	peak_amplitude
        try:
            mpr=peak.pk_name + '_' #we are going to use this a lot so make the variable shorter
            if output_peak_names == True:
                #Then dump full peak names
                logging.debug(str(index) + ' : ' + mpr)
        except Exception as err:
            logging.error(f"Error: {err}. Poorly specified peak?")
            raise        
        
        if pd.isnull(mpr) or pd.isnull(peak.pk_type):
            #Here handle zero values
            logging.warning(f"No input values for peak {index}: name:{mpr}, type:{peak.peak_type},type:{peak.peak_center},type:{peak.peak_sigma},type:{peak.peak_amplitude}")
            continue
      
        if peak.pk_type == 'Gaussian':
            myModel = models.GaussianModel(prefix=mpr)
        elif peak.pk_type == 'Lorentzian':
            myModel = models.LorentzianModel(prefix=mpr)     
        elif peak.pk_type == 'Voigt':
            myModel = models.VoigtModel(prefix=mpr)       
        elif peak.pk_type == 'PseudoVoigt':
            myModel = models.PseudoVoigtModel(prefix=mpr)        
        elif peak.pk_type == 'BreitWignerFano':
            myModel = models.BreitWignerModel(prefix=mpr) 
        elif peak.pk_type == 'BWF_spc':
            myModel = Model(BWF_spc,prefix=mpr) 
        elif peak.pk_type == 'Placeholder':
            logging.info("Add your own peak types here, consult https://lmfit.github.io/lmfit-py/builtin_models.html")
        else:
            logging.warning("No peak type given...skipping")
    
        if index == 0:
            pars = myModel.make_params()
        else:
            pars = pars.update(myModel.make_params()) 
        #allows initial guess amplitude to vary based on data. 'ratiopos': ratio, relative to the height at the first peak position, 'ratiomax': varies as the max height in the function,  "guess" - uses the intensity at the center positions, 'fixed' just use a fixed value
       
        if ((peak.pk_type == 'BreitWignerFano') | (peak.pk_type == 'BWF_spc')):
            myq = peak.pk_par4
        #print(f"Q Value:{myq}")
        #Compute intensities
        if (my_fit_options.amplitude_mode == 'ampratmax') or (my_fit_options.amplitude_mode == 'ampratpos'):
            
            #The initial amplitude is computed using input sigma value, and height as either: 
            # (ratpos): initial center position of the first peak in the list
            # (ratmax): the maximum intensity passed in to the function. 
            #Then all initial guess amplitudes (including the first) 
            # are are as a ratio relative to that value. 
            logging.debug(use_intensity)
            print(use_intensity)

            peak_amplitude = peak.pk_amplitude*height_to_amplitude(use_intensity,my_model_parameters.iloc[0].pk_sigma,curve=my_model_parameters.iloc[0].pk_type,q=myq)
        elif (my_fit_options.amplitude_mode == 'hgtratmax') or (my_fit_options.amplitude_mode == 'hgtratpos'):
            
            #The initial height is computed as either: 
            # (ratpos): initial center position of the first peak in the list
            # (ratmax): the maximum intensity passed in to the function. 
            #Then all initial guess amplitudes (including the first) 
            # are are as a ratio relative to that height, using the guess sigma.
            #This is the only option for which 'peak amplitude' is not technically the correct term
            #Because peak_amplitude value modifies the height, rather than the amplitude. 
            logging.debug(use_intensity)
            peak_amplitude = height_to_amplitude(peak.pk_amplitude*use_intensity,peak.pk_sigma,curve=peak.pk_type,q=myq)
        elif my_fit_options.amplitude_mode == 'apprx':
            #Here we use the intensity at the center position, and compute the amplitude from that number
            peak_amplitude = height_to_amplitude(use_intensity[index],peak.pk_sigma,curve=peak.pk_type,q=myq)
            #use_intensity[index]
        elif my_fit_options.amplitude_mode == 'ratapprx':
            #Here we use the intensity at the center position, and compute the amplitude from that number
            #However, in 'ratapprx', we multiply the amplitude by the value in 'peak_amplitude'
            #this could be used, for exmaple, to rescale if you know that some heights are overpredicting. 
            #Especialy helpful for fits with many peaks. 
            peak_amplitude = peak.pk_amplitude*height_to_amplitude(use_intensity[index],peak.pk_sigma,curve=peak.pk_type,q=myq)
            #use_intensity[index]
        elif my_fit_options.amplitude_mode == 'fixed':
            #Here use the inputted peak_amplitude as the peak height. 
            peak_amplitude = peak.pk_amplitude
        else:
            #Default fixed value
            #Log output: using default peak amplitude.             
            peak_amplitude = 1000
            

        #pdb.set_trace()
        if isValid(peak.pk_center):
            pars[mpr+'center'].set(value=peak.pk_center)
        if isValid(peak.pk_center_min):
            pars[mpr+'center'].set(min=peak.pk_center_min)
        if isValid(peak.pk_center_max):
            pars[mpr+'center'].set(max=peak.pk_center_max)
    
        if isValid(peak.pk_sigma):
            pars[mpr+'sigma'].set(value=peak.pk_sigma)
        if isValid(peak.pk_sigma_min):
            pars[mpr+'sigma'].set(min=peak.pk_sigma_min)
        if isValid(peak.pk_sigma_max):
            pars[mpr+'sigma'].set(max=peak.pk_sigma_max)
            
        if isValid(peak.pk_amplitude):
            pars[mpr+'amplitude'].set(value=peak_amplitude)
        if isValid(peak.pk_amplitude_min):
            pars[mpr+'amplitude'].set(min=peak.pk_amplitude_min)
        if isValid(peak.pk_amplitude_max):
            pars[mpr+'amplitude'].set(max=peak.pk_amplitude_max)
       
        if peak.pk_type == 'Voigt':
            if isValid(peak.pk_par4):
                pars[mpr+'gamma'].set(value=peak.pk_par4)
            if isValid(peak.pk_par4_min):
                pars[mpr+'gamma'].set(min=peak.pk_par4_min)
            if isValid(peak.pk_par4_max):
                pars[mpr+'gamma'].set(max=peak.pk_par4_max)        
    
        if ((peak.pk_type == 'BreitWignerFano') | (peak.pk_type == 'BWF_spc')):
            if isValid(peak.pk_par4):
                pars[mpr+'q'].set(value=peak.pk_par4)
            if isValid(peak.pk_par4_min):
                pars[mpr+'q'].set(min=peak.pk_par4_min)
            if isValid(peak.pk_par4_max):
                pars[mpr+'q'].set(max=peak.pk_par4_max)      
    
        if my_fit_options.method == 'brute':
            if isValid(peak.pk_center_brute_step):
                pars[mpr+'center'].set(brute_step=peak.pk_center_brute_step)                
            if isValid(peak.pk_sigma_brute_step):
                pars[mpr+'sigma'].set(brute_step=peak.pk_sigma_brute_step)
            if isValid(peak.pk_amplitude_brute_step):
                pars[mpr+'amplitude'].set(brute_step=peak.pk_amplitude_brute_step)
    
        if index==0:
            fit_model = myModel
        else:
            fit_model = fit_model + myModel  
    
    
    if my_fit_options.remove_lum == 'Fit':

        model_prefix = 'bk_'
        myModel = models.LinearModel(prefix=model_prefix)
        pars = pars.update(myModel.make_params())
        pars[model_prefix + 'slope'].set(value=0)
        pars[model_prefix + 'intercept'].set(value=0)
  
        fit_model = fit_model + myModel     #Here add a linear fit to the model!
    #import pdb
    #pdb.set_trace() 
    return fit_model,pars

def create_figure(x,y,y_flt,dely,title_spc_id,output_spc_id,output,orig_x,orig_y,comps,residual,my_fit_options,ind_min,ind_max):
    
    #Compute SNR
    #Note: to get SNR this makes some assumptions, namely that
    #       G, D3, D2 all contribute to the 'G peak height and
    #       D1, D3, D4 all contribute to the D peak height
    #       We assume there is always a G and D1 peak
    # If a peak called g_ and d1_ do not exist, then the output will be NaN. 
    #   If the above peaks exist but the other assumptions are in some way invalid, 
    #   This will produce an incorrect result. 
    resi_mean = np.nanmean(residual)
    totdiff = residual-resi_mean
    noise = np.sqrt(np.nansum(totdiff**2)/len(residual))
    
    try:
        comps_g = comps['g_']
        comps_d = comps['d1_']
        
        try:
            comps_g = comps_g + comps['d3_']
            comps_d = comps_d+ comps['d3_']
        except:
            pass
        try:
            comps_g = comps_g + comps['d2_']
        except:
            pass
        try:
            comps_d = comps_d + comps['d4_']
        except:
            pass
        snr_g = np.nanmax(comps_g)/noise
        snr_d = np.nanmax(comps_d)/noise
    except:
        snr_d = np.nan  
        snr_g = np.nan

    snr_text = f'{snr_d:.2f},{snr_g:.2f}'
    
    fig = plt.figure(figsize=(12.8, 14))
    gs = fig.add_gridspec(5,1)
    ax = fig.add_subplot(gs[0:2,:])
    axa = ax.twinx()
    ax.plot([],[],color='k',alpha=0.5,label='orig_data')
    ax.fill_between(x, output.best_fit-dely, output.best_fit+dely, color="#ABABAB",label=r'3-$\sigma$ uncertainty band')
    axa.plot(orig_x,orig_y,alpha=0.5,color='k')
    ax.plot(x,y,label='data')
    ax.plot(x,output.best_fit,label='best fit')
    for mycomp in comps:
        ax.plot(x, comps[mycomp],'--',label=str(mycomp))
    ax.plot(x,residual,'.',label="noise")
    ylims = [np.nanmin(residual),1.4*np.nanmax(y)]    
    ax.set_title(my_fit_options.fit_name + ' ' + title_spc_id)
    t=ax.text(0.02,0.02,snr_text,transform = ax.transAxes,horizontalalignment='left',verticalalignment='bottom')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
    ax.legend(loc='upper left',ncol=1)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xlim([orig_x[ind_min-30],orig_x[ind_max+30]])
    ax.set_ylim(ylims)
    report = str(output.fit_report())
    report_a = report.split('[[Fit Statistics]]')
    model_report = "".join(['      ' + mytxt + '+\n' for mytxt in report_a[0].split('+')]) 
    report_b = report_a[1].split('[[Variables]]')
    report_c = report_b[1].split('[[Correlations]]')
    report_c1 = model_report[0:-2] +'Fit Statistics:'+ report_b[0] 
    report_c3 = "Fit Variables:\n" + "".join([myrep for myrep in report_c[0]])

    if len(report_c) > 1:
        report_c2 =  'Correlations:' + report_c[1]
        ax.text(0.15,-0.05,report_c2,horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    
    ax.text(-0.15,-0.05,report_c1,horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.475,-0.1,report_c3,horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    plt.subplots_adjust(hspace=0) 
    #Sanitize the output file!!
    myspcid = re.sub('[^A-Za-z0-9_-]+','',output_spc_id)
    
    savename = f'{my_fit_options.fit_name}_{myspcid}.png'
    plt.savefig(os.path.join(my_fit_options.fig_output_dir,savename))
    fig1,ax1 = plt.subplots(figsize=(12.8,14))
    ax1.plot(x,output.init_fit,label='init')
    ax1.plot(x,output.best_fit,label='final')
    ax1.plot(x,y,label='data')
    ax1.plot(x,y_flt,label='data-filtered')
    ax1.plot([],[],color=[0.5,0.5,0.5],label='original')
    #ax1.plot([],[],color='k',label='   initipython fit_dg_v6f.py fit_dg_options.csvalconditions')
    ax1a = ax1.twinx()
    ax1a.plot(orig_x,orig_y,color=[0.5,0.5,0.5])
    #ax1a.plot(x,output.init_fit,color='k')
    ax1.set_title('Init:' + my_fit_options.fit_name + ' ' + title_spc_id)
    ax1.legend()
    ax1.set_xlim([orig_x[ind_min-30],orig_x[ind_max+30]])
     
    savename = f'{my_fit_options.fit_name}_{myspcid}_init.png'
    plt.savefig(os.path.join(my_fit_options.fig_output_dir,savename))

 
    plt.close('all')


def dgfit(orig_x,orig_y,orig_wgt,ninputs,my_fit_options,my_model_parameters,input_spc_id,spc_map_pos=None):

    ''' 
    dgfit: this code actually builds the model and fits it. In the older
    version of the code dgfit_worker_txt and dg_fit_worker each had thier own fitting
    text. However, rather than having to duplicate effort and code when 
    modifying the fitting code, better just modify one set of code. 
    '''

    #input_spc_id
    
    if spc_map_pos == None:
        spc_x_pos = 'NaN'
        spc_y_pos = 'NaN'
    else:
        spc_x_pos = spc_map_pos[0]
        spc_y_pos = spc_map_pos[1]

    ind_min=int(my_fit_options.spc_inds[0])
    ind_max=int(my_fit_options.spc_inds[1])

        
    x=orig_x[ind_min:ind_max]
    
    y,baseline_const,baseline_slope=handle_lum(orig_x,orig_y,ind_min,ind_max,remove_lum=my_fit_options.remove_lum, remove_lum_delta= my_fit_options.remove_lum_delta)      

    
    wgt,note,wgt_status=handle_wgt(orig_wgt,ind_min,ind_max,ninputs,my_fit_options.use_weights)

    fit_note_str = ',' + note + ','
    
    #Note: Input_spc_id here is the spectrum id in the case of a large set of spectra in one file
    # In the case of one spectrum per file, it's the path to the file.
    
    #input_spc_id = myfile     
    if my_fit_options.fid_path == False:
        spc_base_id = os.path.basename(input_spc_id)
    else:
        spc_base_id = input_spc_id
        
    spc_short_id = os.path.splitext(spc_base_id)[0] #in the old code this was myfile_basename

    use_intensity = get_init_intensity(x,y,my_model_parameters,my_fit_options.amplitude_mode)
    
    fit_model,pars = build_model(my_fit_options,my_model_parameters,use_intensity=use_intensity)  
    
    fit_input_text = '' 

    for mypar in pars:
        #print(f'{str(mypar)}:{pars[str(mypar)].value}')
        if 'fwhm' in str(mypar):continue
        if 'height' in str(mypar):continue
        if 'dc' in str(mypar):continue

        fit_input_text = fit_input_text + my_fit_options.output_resolution % pars[str(mypar)].value + ',' 

    #Run model
    try:
        print(my_fit_options.method)

        if ((my_fit_options.use_weights==True) & (wgt_status == True)):
            #print("A")
            print(f"using weights: options.use_weights{my_fit_options.use_weights}, wgt_status:{wgt_status}")
            output = fit_model.fit(y, pars, x=x,weights=wgt,scale_covar=False,method=my_fit_options.method)#fit_kws={'factor':myfactor})
        else:
            print("NOT using weights...")
            output = fit_model.fit(y, pars, x=x,method=my_fit_options.method)#fit_kws={'factor':myfactor})
    
        comps = output.eval_components(x=x)
        dely = output.eval_uncertainty(sigma=3)

    except Exception as err:
        fill_value = np.nan
        fit_quality_text =   '{fill_value},{fill_value},{fill_value},{fill_value},{fill_value},{fill_value}'
        fit_result_text = ''
    
        for mypar in pars:
                    if 'fwhm' in str(mypar):continue
                    if 'height' in str(mypar):continue
          
                    fit_result_text = fit_result_text + f'{fill_value},' 
        logging.error(err)
        return_text = "nan,nan,nan," + fit_result_text + fit_quality_text+fit_note_str + fit_input_text+ '\n'
        
        return return_text
    #Process the output results (we have a few additional values we want to putput)
    
    residual = y-output.best_fit
    resi_mean=np.nanmean(residual)
    totdiff = residual-resi_mean
    noise = np.sqrt(np.nansum(totdiff**2)/len(residual))
    mse=np.mean(np.power(residual,2)) 
    chisq_act = output.result.chisqr
    chisq_red = output.result.redchi
    aic = output.result.aic
    bic = output.result.bic
 
    #Create the output_result string
    fit_quality_text =   f'{mse:.1e},{noise:.1e},{chisq_act:.1e},{chisq_red:.1e},{aic:.1e},{bic:.1e}'
    
    fit_result_text = ''
    
    for mypar in pars:
                    if 'fwhm' in str(mypar):continue
                    if 'height' in str(mypar):continue
                    if 'dc' in str(mypar):continue
                    if 'slope' in str(mypar): baseline_slope= output.best_values[str(mypar)]
                    if 'intercept' in str(mypar): baseline_const= output.best_values[str(mypar)]
                     
                    #fit_result_text = fit_result_text + f'{output.best_values[str(mypar)]:.4f},' 
                    try:
                        fit_result_text = fit_result_text + my_fit_options.output_resolution % output.best_values[str(mypar)] + ','  #Too many digits - check your sig figs. 
                    except Exception as err:
                        print(err)
                        pdb.set_trace()
                        raise
    
    my_return_text = f"{spc_base_id},{spc_x_pos},{spc_y_pos},{baseline_const:.4f},{baseline_slope:.4f}," + fit_result_text + fit_quality_text+ fit_note_str+fit_input_text+'\n'
    
    #####
    #
    # Generate the plot
    #
    if my_fit_options.generate_figure==True:
        y_flt = savgol_filter(y,window_length=15,polyorder=2) 
        create_figure(x,y,y_flt,dely,input_spc_id,spc_short_id,output,orig_x,orig_y,comps,residual,my_fit_options,ind_min,ind_max)
    if 'json' in my_fit_options.output_type: #Options: txt, json, txtjson -> json saves every model output to a .json file
        myspcid = re.sub('[^A-Za-z0-9_-]+','',spc_short_id) 
        savename = f'modelresult_{my_fit_options.fit_name}_{myspcid}.json'

        savetxt = os.path.join(my_fit_options.fig_output_dir,savename)
        model.save_modelresult(output,savetxt)



    return my_return_text
    

def dgfit_worker_txt(myfile,my_fit_options,my_model_parameters):
    #Handle the case where individual spectra are in individual data files
    
    #This one is used to generate the figs for Doughty and Hill 2025
    print(myfile) 

    #my_fit_options = deepcopy(fit_options)
    #my_model_parameters = deepcopy(model_parameters)

    data = np.loadtxt(myfile,delimiter=my_fit_options.delimiter,comments=my_fit_options.comments,skiprows=my_fit_options.skiprows,usecols = my_fit_options.usecols)
   
    if len(data) == 0:
        raise NameError(f"Problem loading data file {myfile}")
    
    nrows,ncols=data.shape
    
    if ncols>nrows:
        data = data.transpose()
        nrows,ncols=data.shape
    if ncols >=3:
        orig_wgt =data[:,2]
    else:
        #If no weight specified, ignore. 
        orig_wgt = np.ones(data[:,0].shape)

    orig_x = data[:,0] 
    orig_y = data[:,1] 
    
    return_text = dgfit(orig_x,orig_y,orig_wgt,ncols,my_fit_options,my_model_parameters,myfile)

    return return_text    
    
        
def dgfit_worker(num,dt,xpos,ypos,orig_x,orig_y,orig_wgt,my_fit_options,my_model_parameters):#,p_g,p_d1,p_bk):
    #This code handles when you pass in individual spectral values. 
    

    
    if orig_wgt is not None:
        ninputs=3   
    else:
        ninputs=2
    
    summary_name = f"{num}_{dt}"
    
    return_text = dgfit(orig_x,orig_y,orig_wgt,ninputs,my_fit_options,my_model_parameters,summary_name,[xpos,ypos])

    return return_text
    

def fit_dg_handler():
    #Handles the pre-fitting codes/file outputting, calling etc. 

    my_fit_options = deepcopy(fit_options)
    my_model_parameters = deepcopy(model_parameters)
    
    if not os.path.isdir(my_fit_options.fig_output_dir):
        #Check if output directory exists - create it if it does not exist. 
        os.makedirs(my_fit_options.fig_output_dir)


    if my_fit_options.n_cores > 1:
        my_fit_options.multicore = True

    #Build dummy model to initialize the output file. 
    #***NOT USED FOR ANYTHING - Just used for dummy model to get variables in output file**
    if 'apprx' in my_fit_options.amplitude_mode:
        my_use_intensity = np.full(len(my_model_parameters),1000) #Dummy number here
    elif 'pos' in my_fit_options.amplitude_mode:
        my_use_intensity = 1000
    else:
        my_use_intensity = 1000
    fit_model,pars = build_model(my_fit_options,my_model_parameters,use_intensity=my_use_intensity,output_peak_names=True)    

    init_par_names = ''

    for mypar in pars:
        #print(f'{str(mypar)}:{pars[str(mypar)].value}')
        if 'fwhm' in str(mypar):continue
        if 'height' in str(mypar):continue
        if 'dc' in str(mypar):continue

        init_par_names = init_par_names + 'init_' + str(mypar) + ',' 
    start = datetime.datetime.now()
    
    time_str = start.strftime("%Y%m%d_%H%M")
    
    with open(my_fit_options.fit_name + '.csv', 'w') as f:
        
        f.write('#Output File:' + time_str + ' \n')
        f.write('#' + str(fit_model.left) + '\n')
        f.write('# Fit Method:' + str(my_fit_options.method))
        f.write("# Model Parameters:\n#")
        f.write(model_parameters.to_string(index=False).replace("\n","\n#")) #Dump model parameters
        f.write("\n")
           
        f.write('spc_id,spc_x,spc_y,bkr_c,bkr_s,')
        for mypar in pars:
            if 'fwhm' in str(mypar):continue
            if 'height' in str(mypar):continue
            f.write(f'{str(mypar)},')


        f.write('mse,noise,chisq,chisq_red,aic,bic,note,' + init_par_names + 'placeholder\n') #These are the fit parameters that we will output. 
        print("fit_type:",my_fit_options.spc_type)
        if my_fit_options.file_type=='rdat':
            filename_rdat = my_fit_options.file_base + '.rdat'
            filename_rmta = my_fit_options.file_base + '.rmta'
            rdatdata = np.loadtxt(filename_rdat,delimiter=',')
            dt_meta = pd.read_table(filename_rmta,sep=',')
            wn_values = rdatdata[0,:]
            rs_values = rdatdata[1::,:]

            nrs,jnk = rs_values.shape
            
            if my_fit_options.use_weights == True:      
                #Doing this for now! Really I think we should just provide the weight files. 
                rdatdata_raw = np.loadtxt(my_fit_options.file_rdat_raw,delimiter=',')     
                rs_values_raw_s = rdatdata_raw[1::,:]
                rs_values_raw = process_rs(rs_values_raw_s)
                rs_weights = convert_rs_weight(rs_values_raw)
                list_wgt = list(rs_weights)
            else:
                list_wgt = [None for i in range(nrs)]
            

            #Create list of values to pass in to paralellized code
            list_rs = list(rs_values)
            list_wn = [wn_values for i in range(nrs)]
            
            list_n = list(range(nrs))
            list_dt = dt_meta.DateTime.tolist()
            list_xpos = dt_meta.x.tolist()
            list_ypos = dt_meta.y.tolist()           
        
            if my_fit_options.multicore == True:
                #Check numbeer of computer cores
                n_cores_actual = os.cpu_count()
                if my_fit_options.n_cores > n_cores_actual:
                    logging.error("Incorrect core count specification")
                    raise Exception("You are trying to use {my_fit_options.n_cores}) cores, but there are only {n_cores_actual} cores available")
                else:
                    print(f"Beginning fit using {my_fit_options.n_cores} out of {n_cores_actual} computer cores...")
                    logging.info(f"Beginning fit using {my_fit_options.n_cores} out of {n_cores_actual} computer cores...")
                p = multiprocessing.Pool(my_fit_options.n_cores)
                fit_options_send = [my_fit_options for i in range(len(list_n))]
                model_parameters_send = [my_model_parameters for i in range(len(list_n))]
                for result in p.starmap(dgfit_worker, zip(list_n,list_dt,list_xpos,list_ypos,list_wn,list_rs,list_wgt,fit_options_send,model_parameters_send)):#,list_p_g,list_p_d1,list_p_bk)):
                    f.write(result)
                    f.flush()
            else:
                print("Multicore not selected....")
                logging.info("Multicore note selected....")
                if my_fit_options.spc_type == 'single':
                    i=my_fit_options.n_single
                    result = dgfit_worker(list_n[i],list_dt[i],list_xpos[i],list_ypos[i],list_wn[i],list_rs[i],list_wgt[i],my_fit_options,my_model_parameters)
                    f.write(result)      
                elif my_fit_options.spc_type == 'batch':
                    for i in range(nrs):
                        result = dgfit_worker(list_n[i],list_dt[i],list_xpos[i],list_ypos[i],list_wn[i],list_rs[i],list_wgt[i],my_fit_options,my_model_parameters)
                        f.write(result)

        else: #Here assume one spectrum per file. 
            import glob

            file_basename = my_fit_options.file_base + '*' + my_fit_options.file_type
            my_files= sorted(glob.glob(os.path.join(my_fit_options.target_directory,file_basename)))
            
            if ((my_fit_options.multicore == True) and (len(my_files) > 1)):
                #Check numbeer of computer cores
                n_cores_actual = os.cpu_count()
                if my_fit_options.n_cores > n_cores_actual:
                    logging.error("You are trying to use {my_fit_options.n_cores}) cores, but there are only {n_cores_actual} cores available")         
                    raise Exception("You are trying to use {my_fit_options.n_cores}) cores, but there are only {n_cores_actual} cores available")
                else:
                    logging.info(f"Beginning fit using {my_fit_options.n_cores} out of {n_cores_actual} computer cores...")
                    print(f"Beginning fit using {my_fit_options.n_cores} out of {n_cores_actual} computer cores...")
                p = multiprocessing.Pool(my_fit_options.n_cores)
                #for result in p.map(dgfit_worker_txt, my_files):
                    
                fit_options_send = [my_fit_options for i in range(len(my_files))]
                model_parameters_send = [my_model_parameters for i in range(len(my_files))]
                    
                for result in p.starmap(dgfit_worker_txt, zip(my_files,fit_options_send,model_parameters_send)):
                    f.write(result)
                    f.flush()
            elif len(my_files) > 0:
                logging.info("Multicore not selected, working on multiple files...")
                print("Multicore not selected, working on multiple files...")
                for my_file in my_files:
                    result = dgfit_worker_txt(my_file,my_fit_options,my_model_parameters)
                    f.write(result)     
                
            else:
                logging.info("No files found....")
                print("No files found....")        
    
        ##########################################################################
        ##########################################################################
     
    
        end = datetime.datetime.now()
        print(end-start)
        f.write(f"#{end-start}")
        print("f")
        logging.info("End of fit..")
        #pdb.set_trace()
    logging.info("Generating timestamped input/output file")    
    #Here copy the 
    iostr = my_fit_options.log_name_str + "_input_output.csv"

    with open(iostr,'a') as fo:
        fo.write("Input/Output for fit initialized at: " + my_fit_options.log_datetime_str + "\n")
        fo.write("--------------------------------------------------------------\n")
        fo.write("Input:\n")
        with open(my_fit_options.fit_options_name,'r') as fi:
            fo.write(fi.read())
        fo.write("-------------------------------------------------------------\n")
        fo.write("Build Model Pars (dummy model created in handler):\n")   
        #pdb.set_trace()
        #print(pars)
        fo.write(str(fit_model) + "\n")
        fo.write(lmfit.printfuncs.fit_report(pars))
        fo.write("\n")     
        fo.write("-------------------------------------------------------------\n")
        fo.write("Output:\n")
        with open(my_fit_options.fit_name + '.csv','r') as fi:
            fo.write(fi.read())
        
    
    
    
if __name__=='__main__':
    
    #global fit_options
    #global model_parameters
    
    default_name = "fit_dg_options.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("input",type=str, help="You can add the input file after after the optional parser...", 
                        nargs='?',default=default_name,const=default_name)
    args = parser.parse_args()
    #module_name = 'fit_dg_options' 
    fit_options = fit_options_clean()  
    
    print(args)
    
    fit_file_name = args.input 
    
    logdatetime = time.strftime("%Y%m%d_%H%M%S")
    
    if not os.path.isdir(fit_options.log_dir):
    #Check if output directory exists - create it if it does not exist. 
        os.makedirs(fit_options.log_dir)
    
    log_prefix = os.path.join(fit_options.log_dir,'log_' + Path(fit_file_name).stem + '_')
    
    logging.basicConfig(format='%(asctime)s.%(msecs)03d,%(levelname)s,%(message)s',
                    filename=log_prefix + logdatetime + '.log',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d,%H:%M:%S',
                    force=True) #this last command forces a new logger every t
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.debug(args) 
    #Build the fit optionsn class here.              
    
    fit_options.fit_options_name = fit_file_name
    fit_options.log_name_str = log_prefix + logdatetime
     
    with open(fit_file_name) as f:
        for line in f:
            if line.strip()[0:2] == '##':
                #print(line)
                splitline =line.strip().split(',')
                indicator = splitline[0].strip('#').strip()
                print(indicator,':\t',splitline[1])
                #Why do it this way vs something more elegant?
                #Mainly for simplicity. This way it's harder to accidentally fall into a hole
                #The other reason: all the more 'elegant' ways either potentially cause problems with older versions of python
                #Or they are dangerous (for example creating every input file parameter as an element of the class fit_options. That could alllow for insertion of something malicious if a careless user doesn't check the input file. 
                #
                if indicator == 'method':
                    fit_options.fit_name = splitline[1]
                elif indicator == 'fit_name':
                    fit_options.fit_name = splitline[1]
                elif indicator == 'fig_output_dir':
                    fit_options.fig_output_dir = splitline[1]
                elif indicator == 'fit_notes':
                    fit_options.fit_notes = splitline[1]                
                elif indicator == 'amplitude_mode':
                    fit_options.amplitude_mode = splitline[1]
                elif indicator == 'n_cores':
                    fit_options.n_cores = int(splitline[1])    
                elif indicator == 'output_resolution':
                    fit_options.output_resolution = splitline[1]    
                elif indicator == 'spc_inds':
                    fit_options.spc_inds = [int(splitline[1]), int(splitline[2])]   
                elif indicator == 'remove_lum':
                    fit_options.remove_lum = splitline[1] 
                elif indicator == 'remove_lum_delta':
                    fit_options.remove_lum_delta = int(splitline[1])    
                elif indicator == 'remove_lum_lower': #not implemented
                    fit_options.remove_lum_lower = [int(splitline[1]), int(splitline[2])] 
                elif indicator == 'remove_lum_upper':#not implementd
                    fit_options.remove_lum_upper = [int(splitline[1]), int(splitline[2])] 
                elif indicator == 'spc_type':
                    fit_options.spc_type = splitline[1]  
                elif indicator == 'file_type':
                    fit_options.file_type = splitline[1]  
                elif indicator == 'target_directory':
                    fit_options.target_directory = splitline[1]  
                elif indicator == 'file_base':
                    fit_options.file_base = splitline[1]                  
                elif indicator == 'comments':
                    fit_options.comments = splitline[1]   
                elif indicator == 'output_type':
                    fit_options.output_type = splitline[1]
                elif indicator == 'delimiter':
                    fit_options.delimiter = splitline[1]    
                elif indicator == 'skiprows':
                    fit_options.skiprows= int(splitline[1])    
                elif indicator == 'usecols':
                    if splitline[1] == '':
                        fit_options.usecols= None
                    else:
                        fit_options.usecols = splitline[1]    
                elif indicator == 'use_weights':
                    #:qprint(fit_options.use_weights)
                    fit_options.use_weights = eval_bool(splitline[1]) 
                    #print(splitline[1],eval(splitline[1]),fit_options.use_weights)
                elif indicator == 'file_list_raw':
                    fit_options.file_raw_list = splitline[1]    
                elif indicator == 'generate_figure':
                    fit_options.generate_figure = eval_bool(splitline[1])    
                else:
                    logging.warning(indicator + ":found an invalid input parameter")
     
    #Load the fit model parameters here.                 
    model_parameters = pd.read_csv(fit_file_name,comment='#')     
    #pdb.set_trace()
    model_parameters = model_parameters.dropna(subset=['pk_name']) #Drop NAN rows
    
    fit_dg_handler()





