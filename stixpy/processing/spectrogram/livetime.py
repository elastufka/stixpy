import numpy as np
import pandas as pd
#from astropy.io import fits
#from astropy import constants
import warnings
from datetime import datetime as dt
from datetime import timedelta as td
from .triggergram import Triggergram
from ml_utils import print_arr_stats

def pileup_corr_parameter():
    subc = construct_subcollimator()
    pixel_areas = subc.det.pixel.area
    detector_area = (subc.det.area)[0]
    big_pixel_fraction = pixel_areas[0]/detector_area
    prob_diff_pix = (2./big_pixel_fraction - 1.)/(2./big_pixel_fraction)
    return prob_diff_pix
        
def livetime_fraction(triggergram, det_used, adg_file = '/Users/wheatley/Documents/Solar/STIX/stixpy/stixpy/processing/spectrogram/adg_table.json'):
    adg_sc = pd.read_json(adg_file)
    adg_sc.drop(0, inplace= True) # drop first row
    #adg_sc.reset_index(inplace = True)
    det_select = det_used + 1#np.arange(32) + 1
    #print('det_select',det_select)
    ntrig = triggergram.triggerdata.shape[0]
    tau_array = 10.1e-6 + np.zeros(ntrig) #10.1 microseconds readout time per event
    eta_array = 2.63e-6 + np.zeros(ntrig) #2.63 microseconds latency time per event
    beta = 0.940591 #pileup_corr_parameter() for now
    
    idx_select = [row.ADG_IDX for i,row in adg_sc.iterrows() if row.SC in det_select]
    # these are the agd id needed (1-16)
    # is subcollimator selection wrong? output of IDL command doesn't seem correct, if det_select is
    # supposed to select the adg_idx of the corresponding collimator
    ix_fordet = [list(triggergram.adg_idx).index(i) for i in idx_select] #not exactly value_locate, will need to test
    
    ndt = len(triggergram.t_axis.duration)
    duration = np.tile(triggergram.t_axis.duration, ntrig).reshape((ntrig, ndt))
    tau_rate =  np.tile(tau_array, ndt).reshape((ndt, ntrig)).T / duration
    eta_rate =  np.tile(eta_array, ndt).reshape((ndt, ntrig)).T / duration
    nin = triggergram.triggerdata / (1. -  triggergram.triggerdata * (tau_rate + eta_rate))
    livetime_frac = np.exp( -1 * beta * eta_rate * nin) /(1. + (tau_rate + eta_rate) * nin)
    print('livetime fraction shape',livetime_frac.shape)
    sc_idx = list(adg_sc[adg_sc.SC.isin(det_select)].reset_index().sort_values(by='SC').index.values)
    new_idx = [ix_fordet[idx] for idx in sc_idx]

    if livetime_frac.squeeze().ndim == 1:
        result = livetime_frac[new_idx]
    else:
        print('livetime fraction shape',livetime_frac.shape, len(new_idx))
        result = livetime_frac[new_idx,:] #should be 32xM
        print('result shape',result.shape)
    #result = livetime_fraction[ ix_fordet[sort((where_arr( adg_sc.sc, det_select,/map ))[where_arr( adg_sc.sc, det_select)])], * ]
    
    return result
    
def spectrogram_livetime(spectrogram, level = 4):
    """currently accurate to 1e-3, which does lead to differences with IDL of up to 1 count in the test spectrum"""
    ntimes = spectrogram.n_times#counts.shape[-1]
    nenergies = spectrogram.n_energies
    print('nenergies',nenergies)
    det_used = np.where(spectrogram.detector_mask == 1)[0]
    ndet = det_used.size
    err_low = -1 * spectrogram.trigger_err
    err_none = np.zeros_like(spectrogram.trigger_err)
    err_high = spectrogram.trigger_err
    
    livetime_fracs = []
    for err in [err_low, err_none, err_high]:
        if level == 1:
            dim_counts = (ndet, nenergies, ntimes)
            trig = spectrogram.trigger + err
            #shape0 = ndet#ntimes
            
        elif level == 4:
            dim_counts = (nenergies,ntimes)
            trig = np.transpose((spectrogram.trigger + err) * (np.ones(16)/16.))
            #shape0 = nenergies
        
        if np.sum(np.sign(err))/err.size == -1:
            trig[trig <=0] = 0

        triggergram = Triggergram(trig, spectrogram.t_axis)
        livetime_frac = livetime_fraction(triggergram, det_used)
        if level == 4:
            livetime_frac = livetime_frac[0,:]
        livetime_frac = np.tile(livetime_frac,nenergies).reshape(dim_counts) #formerly tile by shape0
        if level == 4:
            livetime_frac = livetime_frac.T
        print('livetime_frac')
        print_arr_stats(livetime_frac)
        #livetime_frac
#        Shape: (30, 16213)
#        Mean: 0.9834991069880434
#        Min: 0.35772757294071816
#        Max: 0.9991485993592356
#        Std: 0.06340875453764692
        livetime_fracs.append(livetime_frac)
    
    if level not in [1,4]:
        warnings.warn('Currently supported compaction levels are 1 (pixel data) and 4 (spectrogram)')
    spec_counts = spectrogram.counts.copy()
    corrected_counts_lower =  spec_counts/livetime_fracs[0]
    corrected_counts =  spec_counts/livetime_fracs[1]
    corrected_counts_upper =  spec_counts/livetime_fracs[2]
    print("corrected low")
    print_arr_stats(corrected_counts_lower)
    print("corrected high")
    print_arr_stats(corrected_counts_upper)
    error_from_livetime = (corrected_counts_upper - corrected_counts_lower)/2.
    print("error from livetime")
    print_arr_stats(error_from_livetime)
    temp_err = spectrogram.error.copy()#np.zeros_like(spectrogram.error.T) #should probably not be zeros at this point... mention to Ewan
    if temp_err.shape != livetime_fracs[1].shape:
        temp_err = spectrogram.error.T#np.zeros_like(spectrogram.error.T) for testing only!
    print_arr_stats(temp_err)
#    Shape: (30, 16213)
#    Mean: 12.629071235656738
#    Min: 0.0
#    Max: 2381.22412109375
#    Std: 59.45590591430664
    
    corrected_error = np.sqrt((temp_err/livetime_fracs[1])**2. + error_from_livetime**2.)

    return corrected_counts, corrected_error, livetime_fracs[1]
    
#livetime low
#Shape: (30, 16213)
#Mean: 0.9999967141537821
#Min: 0.9987837192545421
#Max: 1.0
#Std: 4.959035033820174e-05
#
#livetime high
#Shape: (30, 16213)
#Mean: -0.7421604827722543
#Min: -172.6089747330413
#Max: 0.9989661588652159
#Std: 13.804812776136128
#
#error from livetime
#Shape: (30, 16213)
#Mean: -65.07533764601018
#Min: -39223.14208394954
#Max: 13097.376256984862
#Std: 1142.0640127784686

#corrected low
#Shape: (30, 16213)
#Mean: 274.2012203850798
#Min: 0.0
#Max: 77994.26313052507
#Std: 2146.8548520136537
#
#corrected high
#Shape: (30, 16213)
#Mean: 144.0505450930595
#Min: -11559.565138109288
#Max: 36969.41197089473
#Std: 834.3900972579378
