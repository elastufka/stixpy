import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy import constants
import warnings
from datetime import datetime as dt
from datetime import timedelta as td

from .read_elut import read_elut
from .spectrogram import Spectrogram
from .triggergram import Triggergram
from .spectrogram_axes import stx_time_axis, stx_energy_axis
from .spectrogram_utils import *
from .convert_spectrogram import *
from matplotlib import pyplot as plt
from ml_utils import print_arr_stats
import pidly
import os

def get_l4_testfiles():
    l4_test  ="/Users/wheatley/Documents/Solar/STIX/single_event/220430/solo_L1_stix-sci-xray-spec_20220430T144000-20220430T204502_V01_2204309235-49292.fits"
    bg_file = "/Users/wheatley/Documents/Solar/STIX/single_event/220430/solo_L1A_stix-sci-xray-l1-2204299453_20220429T202959-20220429T211459_058943_V01.fits"
    return l4_test, bg_file

def get_l1_testfiles():
    l1pix = "/Users/wheatley/Documents/Solar/STIX/single_event/220806/solo_L1A_stix-sci-xray-l1-2208046494_20220804T132609-20220804T134138_080909_V01.fits"
    bg_file = "/Users/wheatley/Documents/Solar/STIX/single_event/220806/solo_L1A_stix-sci-xray-l1-2207295463_20220729T083652-20220729T092452_079271_V01.fits"
    return l1pix, bg_file

def mean_minmax(arr):
    vals = list(arr.shape)
    vals.extend([np.mean(arr),np.min(arr),np.max(arr)])
    return vals
    
def test_config(atol = 1e-4, background = False, pixel = False, energy_shift = 0, use_discriminators = True, replace_doubles = False, keep_short_bins = True, shift_duration = None, alpha = None, time_bin_filename = None):
    return locals()
    
def test_l4_from_fits(**kwargs):
    l4_test, _ = get_l4_testfiles()
    test_from_fits(l4_test, **kwargs)
    
def test_l1_from_fits(**kwargs):
    l1_test, _ = get_l1_testfiles()
    kwargs['pixel'] = True
    test_from_fits(l1_test, **kwargs)
    
#def test_l1a_from_fits(atol = 1e-4,energy_shift = 0, use_discriminators = True, replace_doubles = False, keep_short_bins = True, shift_duration = None, alpha = None, time_bin_filename = None):
#    l4_test, _ = get_l4_testfiles(**kwargs)
#    test_from_fits(l4_test)
    
def test_l1bg_from_fits(**kwargs):
    _, l1bg_test = get_l1_testfiles()
    kwargs['pixel'] = True
    kwargs['background'] = True
    kwargs['use_discriminators'] = False
    test_from_fits(l1bg_test, **kwargs)
    
def test_from_fits(fitsfile, background = False, pixel=False, atol = 1e-4,energy_shift = 0, use_discriminators = True, replace_doubles = False, keep_short_bins = True, shift_duration = None, alpha = None, time_bin_filename = None):
    spec = Spectrogram(fitsfile, background = background)
    
    ## same in IDL
    idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl("add_path, '/Users/wheatley/Documents/Solar/STIX/code/STIX-GSW/' + get_delim() +'stix', /expand")
    idl(".compile mrdfits")
    idl("spec", fitsfile)
    idl("energy_shift", energy_shift)
    #if spec.alpha:
    #    idl("alpha", 1)
    if not use_discriminators:
        idl("use_discriminators", 0) #defalut is 1
    else:
        idl("use_discriminators", 1)
    if replace_doubles:
        idl("replace_doubles", 1)
    else:
        idl("replace_doubles",0)
    if keep_short_bins:
        idl("keep_short_bins", 1)
    else:
        idl("keep_short_bins",0)
    #idl("bkspec", bg_file)
    if not pixel:
        idl("stx_read_spectrogram_fits_file, spec, 0., primary_header = primary_header, data_str = data_str, data_header = data_header, control_str = control_str, control_header= control_header, energy_str = energy_str, energy_header = energy_header, t_axis = t_axis, energy_shift = energy_shift,  e_axis = e_axis , use_discriminators = use_discriminators, replace_doubles = replace_doubles, keep_short_bins = keep_short_bins")
    else:
        idl("stx_read_pixel_data_fits_file, spec, 0., primary_header = primary_header, data_str = data_str, data_header = data_header, control_str = control_str, control_header= control_header, energy_str = energy_str, energy_header = energy_header, t_axis = t_axis, energy_shift = energy_shift,  e_axis = e_axis , use_discriminators = use_discriminators, alpha = alpha")

    ## compare counts, triggers, and errors
    idl("counts = data_str.counts")
    idl("counts_err = data_str.counts_err")
    idl("triggers = data_str.triggers")
    idl("triggers_err = data_str.triggers_err")
    idl_counts = idl.counts
    idl_counts_err = idl.counts_err
    idl_triggers = idl.triggers
    idl_triggers_err = idl.triggers_err
    
    if background:
        assert_allclose(spec.counts[0], idl_counts, atol=atol)
        assert_allclose(spec.counts_err[0], idl_counts_err, atol=atol)
        assert_allclose(spec.triggers[0], idl_triggers, atol=atol)
        assert_allclose(spec.triggers_err[0], idl_triggers_err, atol=atol)
    else:
        assert_allclose(spec.counts, idl_counts, atol=atol)
        assert_allclose(spec.counts_err, idl_counts_err, atol=atol)
        assert_allclose(spec.triggers, idl_triggers, atol=atol)
        assert_allclose(spec.triggers_err, idl_triggers_err, atol=atol)
    
    ## compare time_bin_center and duration
    idl("time = data_str.time")
    idl("timedel = data_str.timedel")
    idl_time_bin_center = idl.time
    idl_duration = idl.timedel
    assert_allclose(spec.time_bin_center, idl_time_bin_center, atol=atol)
    assert_allclose(spec.t_axis.duration, idl_duration, atol=atol)
    
    ## compare energy axis
    idl("energy_mean = e_axis.mean")
    idl("energy_gmean = e_axis.gmean")
    idl("energy_low = e_axis.low")
    idl("energy_high = e_axis.high")
    idl("energy_width = e_axis.width")
    idl("energy_low_fsw_idx = e_axis.low_fsw_idx")
    idl("energy_high_fsw_idx = e_axis.high_fsw_idx")
    idl_energy_mean = idl.energy_mean
    idl_energy_gmean = idl.energy_gmean
    idl_energy_low = idl.energy_low
    idl_energy_high = idl.energy_high
    idl_energy_width = idl.energy_width
    idl_energy_low_fsw_idx = idl.energy_low_fsw_idx
    idl_energy_high_fsw_idx = idl.energy_high_fsw_idx
    idl.close()
    assert_allclose(spec.e_axis.energy_mean, idl_energy_mean, atol=atol)
    assert_allclose(spec.e_axis.gmean, idl_energy_gmean, atol=atol)
    assert_allclose(spec.e_axis.low, idl_energy_low, atol=atol)
    assert_allclose(spec.e_axis.high, idl_energy_high, atol=atol)
    assert_allclose(spec.e_axis.width, idl_energy_width, atol=atol)
    assert_allclose(spec.e_axis.low_fsw_idx+1, idl_energy_low_fsw_idx, atol=atol)
    assert_allclose(spec.e_axis.high_fsw_idx+1, idl_energy_high_fsw_idx, atol=atol)

    print("test passed successfully")
    
def test_l4_initialization(**kwargs):
    l4_test, _ = get_l4_testfiles()
    test_initialization(l4_test, **kwargs)
    
def test_l1_initialization(**kwargs):
    l1_test, _ = get_l1_testfiles()
    kwargs['pixel'] = True
    test_initialization(l1_test, **kwargs)
    
#def test_l1a_from_fits(atol = 1e-4,energy_shift = 0, use_discriminators = True, replace_doubles = False, keep_short_bins = True, shift_duration = None, alpha = None, time_bin_filename = None):
#    l4_test, _ = get_l4_testfiles(**kwargs)
#    test_from_fits(l4_test)
    
def test_l1bg_initializatino(**kwargs):
    _, l1bg_test = get_l1_testfiles()
    kwargs['pixel'] = True
    kwargs['background'] = True
    kwargs['use_discriminators'] = False
    test_initialization(l1bg_test, **kwargs)
    
def test_initialization(fitsfile, background = False, pixel=False, atol = 1e-4,energy_shift = 0, use_discriminators = True, replace_doubles = False, keep_short_bins = True, shift_duration = None, alpha = None, time_bin_filename = None):
    spec = Spectrogram(fitsfile, background = background)
    spec.initialize_spectrogram()

    ## same in IDL
    #os.chdir('/Users/wheatley/Documents/Solar/STIX/code/STIX-GSW/')
    idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl("add_path, '/Users/wheatley/Documents/Solar/STIX/code/STIX-GSW/' + get_delim() +'stix', /expand")
    idl("setenv, 'SSW_STIX=/Users/wheatley/Documents/Solar/STIX/code/STIX-GSW/stix'")
    idl(".compile mrdfits")
    #idl("cd, '/Users/wheatley/Documents/Solar/STIX/code/STIX-GSW'")
    idl("fits_path_data", fitsfile)
    idl("energy_shift", energy_shift)
    #if spec.alpha:
    #    idl("alpha", 1)
    if not use_discriminators:
        idl("use_discriminators", 0) #defalut is 1
    else:
        idl("use_discriminators", 1)
    if replace_doubles:
        idl("replace_doubles", 1)
    else:
        idl("replace_doubles",0)
    if keep_short_bins:
        idl("keep_short_bins", 1)
    else:
        idl("keep_short_bins",0)
    #idl("bkspec", bg_file)
    if not pixel:
        idl("stx_convert_spectrogram_test, fits_path_data=fits_path_data, fits_path_bk = fits_path_bk, time_shift=0., energy_shift = energy_shift, distance = distance, flare_location= flare_location, replace_doubles = replace_doubles, keep_short_bins = keep_short_bins, apply_time_shift = apply_time_shift, shift_duration = shift_duration, no_attenuation=no_attenuation, sys_uncert = sys_uncert, generate_fits = generate_fits, specfile = specfile, srmfile = srmfile, xspec = xspec, background_data = background_data, plot = plot, ospex_obj = ospex_obj, spectrogram = spectrogram")
    else:
        idl("stx_convert_pixel_data_test, fits_path_data=fits_path_data, fits_path_bk = fits_path_bk, time_shift=0., energy_shift = energy_shift, distance = distance, flare_location= flare_location, shift_duration = shift_duration, no_attenuation=no_attenuation, sys_uncert = sys_uncert, generate_fits = generate_fits, specfile = specfile, srmfile = srmfile, xspec = xspec, background_data = background_data, plot = plot, ospex_obj = ospex_obj, spectrogram = spectrogram, eff_ewidth = eff_ewidth")

    ## compare counts, triggers, and errors
    idl("counts = spectrogram.counts")
    idl("counts_err =  spectrogram.error")
    idl("trigger = spectrogram.trigger")
    idl("trigger_err = spectrogram.trigger_err")
    idl_counts = idl.counts
    idl_counts_err = idl.counts_err
    idl_trigger = idl.trigger
    idl_trigger_err = idl.trigger_err

#    if background:
#        assert_allclose(spec.counts[0], idl_counts, atol=atol)
#        assert_allclose(spec.counts_err[0], idl_counts_err, atol=atol)
#        assert_allclose(spec.trigger[0], idl_trigger, atol=atol)
#        assert_allclose(spec.trigger_err[0], idl_trigger_err, atol=atol)
#    else:
#        assert_allclose(spec.counts.T, idl_counts, atol=atol)
#        #assert_allclose(spec.counts_err, idl_counts_err, atol=atol)
#        assert_allclose(spec.trigger.T, idl_trigger, atol=atol)
#        assert_allclose(spec.trigger_err.T, idl_trigger_err, atol=atol)
        
    if pixel:
        idl_eff_ewidth = idl.eff_ewidth
        assert_allclose(spec.eff_ewidth, idl_eff_ewidth, atol=atol)

    idl.close()
    print("test passed successfully")

def test_convert_l4(atol = 1e-4):
    l4_test, bg_file = get_l4_testfiles()
    rdict = convert_spectrogram(l4_test, bg_file)
    spec = rdict['spec']
    spec_bk = rdict['spec_bk']
    
#    corrected_counts_bk          30       16213
#    Mean:        30.542080
#    Min:       0.61343993
#    Max:        2367.1158
#    Std:        97.374006
    print('corrected_counts_bk',np.allclose(mean_minmax(rdict['corrected_counts_bk']),[30,16213,30.542080,0.61343993,2367.1158],atol=atol))
#    spec_In_bk          30       16213
#    Mean:        30.512577
#    Min:       0.61284747
#    Max:        2364.8362
#    Std:        97.279997
    print('spec_in_bk',np.allclose(mean_minmax(rdict['spec_in_bk']),[30,16213,30.512577,0.61284747,2364.8362],atol=atol))
#    error_bk          30       16213
#    Mean:        5.3090429
#    Min:       0.55412221
#    Max:        223.33762
#    Std:        10.486871
    print('error_bk',np.allclose(mean_minmax(rdict['error_bk']),[30,16213,5.3090429,0.55412221,223.33762],atol=atol))
#    corrected_counts          30       16213
#           215.27700       366.47204       496.66668       303.65831       150.83012       49.267421
#           32.883653       26.032736       20.109790       7.9740476       10.027424
#    Mean:        387.82074
#    Min:        0.0000000
#    Max:        218026.97
#    Std:        4105.4163
#
    print('corrected_counts',np.allclose(mean_minmax(rdict['corrected_counts']),[30,16213,387.82074,0.0000000,218026.97],atol=atol))

#    corrected_error          30       16213
#    Mean:        103.20699
#    Min:        0.0000000
#    Max:        39223.142
#    Std:        1139.2524
    print('corrected_error',np.allclose(mean_minmax(rdict['corrected_error']),[30,16213,103.20699,0.0000000,39223.142],atol=atol))
    
    #IDL
#    spec_in_corr
#    Mean:        243.87446
#    Min:       -788.57357
#    Max:        77993.291
#    Std:        2147.5142


    print('spec_in_corr',np.allclose(mean_minmax(rdict['spec_in_corr']),[30,16213,243.87446,-788.57357,77993.291],atol=atol))
    print(mean_minmax(rdict['spec_in_corr']))
#    spec_in_uncorr
#    Mean:        243.68858
#    Min:       -788.58983
#    Max:        77991.548
#    Std:        2147.4087
    print('spec_in_uncorr',np.allclose(mean_minmax(rdict['spec_in_uncorr']),[30,16213,243.68858,-788.58983,77991.548],atol=atol))
#    total_error
#    Mean:        76.798554
#    Min:       0.43368515
#    Max:        18987.020
    print('total_error',np.allclose(mean_minmax(rdict['total_error']),[30,16213,76.798554, 0.43368515,18987.020],atol=atol))
    
