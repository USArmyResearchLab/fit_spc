
# fit_spc:  Fitting sets of functions to many spectra

Distribution Statement A: Approved for public release: distribution is unlimited


10 February 2025 - Version 1.0.

Code by David C. Doughty<sup>1</sup> and Steven C. Hill<sup>1 </sup>

[1] US Army DEVCOM Army Research Laboratory

 
 
## Motivation

---

This software allows for the batch fitting of functions to spectra. The focus here is on Raman spectroscopy, with data either consisting of individual files containing spectra or a single file containing multiple spectra. In principle it could be applied to any 1d spectral data. 

The second goal of this code is to reproduce the results in Doughty and Hill 2026. We (Doughty and Hill) also believe that the code and data used to do science should be made easily available after publication of a paper. This way others can easily use it for their own research, and also potentially detect bugs or problems in the software.


## Dependencies

---

To run the software, you need an installation of Python 3. It is recommended to use [Anaconda](https://www.anaconda.com/distribution/) because it is easier to manage required packages. Packages needed:

* matplotlib
* numpy
* pandas
* scipy (and their respective dependencies)

You will also need to install the [LMFIT package](https://lmfit.github.io/lmfit-py/). 

All the python codes except fit_spc.py were run on a Windows system, and are used to generate figures for Doughty and Hill (2025). 

The fit_spc.py code was run on a linux system using the runall.sh bash script to run through each configuration file. 

## Installing and Configuration

---

Unzip the file or pull the repo from the git server. You will also need to pull the current rs_tools.py from [our Github Repository](https://github.com/USArmyResearchLab/raman_spec_tools)

Sample input files are found in fit_configuration
Data files are "selected*.txt". 

The main code is fit_spc.py.

* data_files/ contains the data files for the paper
* fit_configuration/ contains the fit configuration files 
* fit_results/ contains the output results from a previous run (in order to replicate the results of Doughty and Hill (2026). Note that running the fitting code will place the results in the current directory
* logs/ is an empty directory where the logs will be placed. 

## Executing the code

The code can be executed using the command
```
python fit_spc.py [configuration file] 
```
from the linux command line. Detailed specific instructions on configuration are provided below

### Configuration file

---

The configuration file has two main sections: the options, and the line shapes. 

The 'options' lines start with "##" followed by a space, the option to be set, and then any option values. 

Available options are described in the example options files here. Consult those files for more information. To generate with weights assuming shot noise, change use_weights to TRUE.

After these lines are several comment lines, and then the peak names, types, and initial conditions for each peak are added. 
```
pk_name,pk_type,pk_center,pk_sigma,pk_amplitude,pk_par4,pk_center_min,pk_center_max......
g,BWF_spc,1600,35,0.25,-50,,,0.01,,0,,,,,,,
d1,Lorentzian,1330,87.5,0.25,,,,0.01,,0,,,,,,, 
```
* The first column (pk_name) here describes the peak name, and that name will be appended in the output file. 
* The second column (pk_type) describes the model function used. Options available include ``Lorentzian``, ``Gaussian``, ``BreitWignerFano``, ``BWF_spc``, and ``Voight``. 
* The third column (pk_center) describes the initial peak center. This should be in whatever the units of the input file are
* The fourth column describes the initial peak sigma (related to width). Consult lmfit guide for definition of sigma.
* The fifth column describes the initial peak amplitude. This value has slightly different behavior depending on what option is selected for the option "amplitude_mode". If "apprx" is selected it will approximate this value based on the initial sigma, and the smoothed value of the input file at peak_center. In this case, pk_amplitude does nothing. If "ratapprx" is selected, then the code will approximate the peak amplitude based on the initial sigma, and some fraction of the smoothed value of the input function at peak_center. That fraction is the value here. If "fixed" is used then whatever value is entered here is used as the initial peak height. 
* The sixth column (pk_par4) is only used with four-parameter fits (BWF or Voight). This is the initial value of that parameter
* The next columns allow you to set bounds on the possible values an output function can take.[The implementation is discussed here](https://lmfit.github.io/lmfit-py/bounds.html#bounds-chapter). Generally it is recommended to not set these if possible. However, generally it is recommended to limit the amplitude_min to zero. 
* The final columns describe parameters used in a brute force fit (this brute-force scans over the possible function space)


### Automatically running all the data files for the paper

---


The software runall.sh will loop through all configuration files over all input spectra if run on a compatible system. This may take a while. Note that this outputs the fit results to the main directory. 
The outputs from runall.sh have been stored in fit_results. These are provided for convenience of reviewers and will probably be removed at a later date. In some cases github clobbers the shebang line on this script. 

To fix (from https://unix.stackexchange.com/questions/27054/bin-bash-no-such-file-or-directory): 
```
:set ff=unix
:set nobomb
:wq
```

The codes starting with "Generate*" make the figures for the Doughty and Hill paper [submitted](https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-m1tfz). 

### Multicore

---

The code now allows for multiple cores to be run. Modify the n_cores variable to set the number of cores used. 





