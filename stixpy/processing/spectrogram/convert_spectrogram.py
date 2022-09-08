import numpy as np
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
from matplotlib import pyplot as plt
from ml_utils import print_arr_stats

#def read_spectrogram_fits_file(fits_path, background = False, time_shift = 0, energy_shift = 0, use_discriminators = True, replace_doubles = False, keep_short_bins = True, shift_duration = None, alpha = 0, time_bin_filename = None): #data_str = None, data_header = None, control_str = None, control_header= None, energy_str = None, energy_header = None, t_axis = None, e_axis = None ,
#    """Read spectrogram FITS file
#    args:
#
#    fits_path_data : str
#    The path to the sci-xray-spec (or sci-spectrogram) observation file
#
#    kwargs:
#
#    background : bool, default = False
#    Is the input file a background file or not
#
#    time_shift : optional, float, default=0.
#    The difference in seconds in light travel time between the Sun and Earth and the Sun and Solar Orbiter i.e. Time(Sun to Earth) - Time(Sun to S/C)
#
#    energy_shift : optional, float, default=0.
#    Shift all energies by this value in keV. Rarely needed only for cases where there is a significant shift in calibration before a new ELUT can be uploaded.
#
#    alpha : bool, default=0
#    Set if input file is an alpha e.g. L1A
#
#    use_discriminators : bool, default=False
#    an output float value
#
#    shift_duration : None
#    Shift all time bins by 1 to account for FSW time input discrepancy prior to 09-Dec-2021. N.B. WILL ONLY WORK WITH FULL TIME RESOLUTION DATA WHICH IS USUALLY NOT THE CASE FOR SPECTROGRAM DATA.
#     """
#
#    short_bins_dt = dt.strptime('2020-11-25T00:00:00',"%Y-%m-%dT%H:%M:%S")
#    shift_duration_dt = dt.strptime('2021-12-09T00:00:00',"%Y-%m-%dT%H:%M:%S")
#
#    hdul = fits.open(fits_path)
#    primary_header = hdul[0].header
#    control = hdul[1]
#    data = hdul[2]
#    energy = hdul[3] if hdul[3].name == 'ENERGIES' else hdul[4]
#    data_header = data.header
#    shift_step = 0
#
#    n_time = data.data['time'].size
#
#    energies_used = np.where(control.data.energy_bin_mask == 1)[1]
#    #nenergies = energies_used.size
#    processing_level = primary_header['LEVEL']
#    if processing_level.strip() == 'L1A' and alpha is None: alpha = 1
#    else:
#        alpha = 0
#
#    try:
#        hstart_str = primary_header['DATE_BEG']
#    except KeyError:
#        hstart_str = primary_header['DATE-BEG']
#    hstart_time = dt.strptime(hstart_str,"%Y-%m-%dT%H:%M:%S.%f")
#    print(f"hstart_time {hstart_time}")
#
#    #trigger_zero should always be 0 as far as I know... it gets modified by mreadfits 2.26
##    try:
##        trigger_zero = data.header['TZERO3']
##    except KeyError:
##        trigger_zero = 0
##    new_triggers = np.array(trigger_zero) + data.data.triggers # 1D array
##    data.data.triggers[:] = new_triggers
###    except KeyError:
###        pass
#
#
#    data.data.counts_err = np.sqrt(data.data.counts_err**2 + data.data.counts)
#    data.data.triggers_err = np.sqrt(data.data.triggers_err**2 + data.data.triggers)
#
#    if hstart_time < dt.strptime('2021-12-09T00:00:00',"%Y-%m-%dT%H:%M:%S"):
#        shift_duration = 1
#
#    #If time range of observation is during Nov 2020 RSCW apply average energy shift by default
#    if hstart_time > dt.strptime('2020-11-15T00:00:00',"%Y-%m-%dT%H:%M:%S") and hstart_time < dt.strptime('2021-12-04T00:00:00',"%Y-%m-%dT%H:%M:%S"):
#        energy_shift = 1.6
#        warnings.warn('Warning: Due to the energy calibration in the selected observation time a shift of -1.6 keV has been applied to all science energy bins.')
#
#    if not keep_short_bins and hstart_time < short_bins_dt:
#        warnings.warn('Automatic short bin removal should not be attempted on observations before 25-Nov-20')
#
#    if shift_duration is not None and hstart_time > shift_duration_dt:
#        warnings.warn('Shift of duration with respect to time bins is no longer needed after 09-Dec-21')
#
#    if shift_duration is not None and hstart_time < shift_duration_dt: # Shift counts and triggers by one time step - for use in background file?
#        shift_step = -1 #otherwise default is zero and nothing happens
#
#    print('shift step', shift_step)
#    if background:
#        axis = -1 #time axis is last now...need to test this
#    else:
#        axis = 0
#    counts = shift_one_timestep(data.data.counts, shift_step = shift_step, axis = axis)
#    counts_err = shift_one_timestep(data.data.counts_err, shift_step = shift_step, axis = axis)
#    triggers = shift_one_timestep(data.data.triggers, shift_step = shift_step, axis = axis)
#    triggers_err = shift_one_timestep(data.data.triggers_err, shift_step = shift_step, axis = axis)
#    duration = shift_one_timestep(data.data.timedel, shift_step = -1*shift_step)
#    time_bin_center = shift_one_timestep(data.data.time, shift_step = -1*shift_step)
#    control_index = shift_one_timestep(data.data.control_index, shift_step = -1*shift_step)
#
#    if not keep_short_bins:
#        # Remove short time bins with low counts
#        counts_for_time_bin = sum(counts[1:10,:],1) # shape is ex. (32,)
#        idx_short = np.where(counts_for_time_bin >= 1400)[0]
#
#        # List when we have 1 second time bins, short or normal bins
#        mask_long_bins = np.ones(n_time-1)
#
#        min_time_table = pd.read_csv(f"{os.environ['STX_DET']}/min_time_index.csv")
#
#        min_time = df.where(hstart_str < df.where(hstart_str > df[' start_date'])[' end_date']).dropna(how='all')['Mininmum time [cs]'].values[0]/10.
#        print(f"min_time {min_time}")
#
#        if idx_short.size > 0:
#            idx_double = np.where(duration[idx_short] == min_time)[0]
#            if idx_double.size > 0:
#                idx_short_plus = [idx_short, idx_short[idx_double]-1] #check
#            else:
#                idx_short_plus = idx_short
#
#        if idx_double.size > 0 and replace_doubles:
#            mask_long_bins[idx_short] = 0
#            duration[idx_short[idx_double]-1] = (duration[max(np.where(duration[0:idx_short[idx_double]] > 1))] + duration[min(np.where(duration[idx_short[idx_double]:-1] > 1)) + idx_short[idx_double]])/2.
#        else:
#            mask_long_bins[idx_short_plus] = 0
#
#        idx_long = np.where(mask_long_bins == 1)[0]
#
#        time_bin_center = time_bin_center[idx_long]
#        duration = duration[idx_long]
#        counts = counts[idx_long,:]
#        counts_err = counts_err[idx_long,:]
#        triggers = triggers[idx_long]
#        triggers_err = triggers_err[idx_long]
#
#    n_time = time_bin_center.size
#
#    rcr = data.data.rcr # byte array
#    if alpha: # things specific to L1A files
#        try:
#            rcr = control.data.rcr #need to reshape?
#        except AttributeError:
#            pass
#
#        full_counts = np.zeros((n_time,32))
#        full_counts[:, energies_used] = counts
#        counts = full_counts
#
#        full_counts_err = np.zeros((n_time, 32))
#        full_counts[:,energies_used] = counts_err
#        counts_err = full_counts_err
#
#    spec_data = {'time': time_bin_center,
#      'timedel': duration,
#      'triggers': triggers,
#      'triggers_err': triggers_err,
#      'counts': counts,
#      'counts_err': counts_err,
#      'control_index': control_index,
#      'rcr': rcr,
#      'header': data_header}
#
#    if 'pixel_masks' in control.data.names:
#        spec_data['pixel_mask'] = control.data.pixel_masks
#        spec_data['detector_mask'] = control.data.detector_masks
#    elif 'pixel_mask' in control.data.names:
#        spec_data['pixel_mask'] = control.data.pixel_mask
#        spec_data['detector_mask'] = control.data.detector_mask
#    else:
#        spec_data['pixel_mask'] = data.data.pixel_masks
#        spec_data['detector_mask'] = data.data.detector_masks
#
#    #check shape
##    if spec_data['pixel_mask'].size !=12:
##       spec_data['pixel_mask'] = np.sum(spec_data['pixel_mask'], axis =1)
##    if spec_data['detector_mask'].size !=32:
##        nelem = spec_data['detector_mask'].shape[0]
##        spec_data['detector_mask'] = np.sum(spec_data['detector_mask'], axis =0)/nelem
#
#    if background:
#        spec_data['pixel_mask'] = np.ones((1,12)) #will be changed in convert_spectrogram later
#        spec_data['detector_mask'] = np.ones((1,32))
#        spec_data['num_pixel_sets'] = data.data.num_pixel_sets
#        spec_data['num_energy_groups'] = data.data.num_energy_groups
#
#    # Create time axis
#    #TUNIT1 is Time unit
#    if data_header['TUNIT1'].strip() == 's':
#        factor = 1
#    elif data_header['TUNIT1'].strip() == 'cs':
#        factor = 100
#
#    start_time = hstart_time + td(seconds = time_shift)
#    t_start = [start_time + td(seconds = bc/factor - d/(2.*factor)) for bc,d in zip(time_bin_center, duration)]
#    t_end = [start_time + td(seconds = bc/factor + d/(2.*factor)) for bc,d in zip(time_bin_center, duration)]
#    t_mean = [start_time + td(seconds = bc/factor) for bc in time_bin_center]
#    t_axis = stx_time_axis(time_mean = t_mean, time_start = t_start, time_end = t_end, duration = duration/factor)
#
#    if (control.data.energy_bin_mask[0][0] or control.data.energy_bin_mask[0][-1]) and use_discriminators:
#        control.data.energy_bin_mask[0][0] = 0
#        control.data.energy_bin_mask[0][-1] = 0
#        print("after use discriminators", spec_data['counts'].shape)
#        spec_data['counts'][...,0] = 0. #originally [0,:]
#        spec_data['counts'][...,-1] = 0.
#        spec_data['counts_err'][...,0] = 0.
#        spec_data['counts_err'][...,-1] = 0.
#
#    energies_used = np.where(control.data.energy_bin_mask == 1)[1]
#    #nenergies = energies_used.size
#    print(f"energy bin mask {control.data.energy_bin_mask}")
#    print(f"energies_used {energies_used}")
#    energy_edges_2 = np.transpose([[energy.data.e_low[energies_used]], [energy.data.e_high[energies_used]]])
#    _, _, _, energy_edges_1, _ = edge_products(energy_edges_2.squeeze())
#    print(f"energy_edges_1 {energy_edges_1.size}")
#
#    energy_edges_all2 = np.transpose([[energy.data.e_low], [energy.data.e_high]])
#    _, _, _, energy_edges_all1, _ = edge_products(energy_edges_2.squeeze())
#    print(f"energy_edges_all1 {energy_edges_all1.size}")
#
#    #remove infs
#    #energy_edges_all1 = energy_edges_all1[~np.isinf(energy_edges_all1)]
#    #energy_edges_1 = energy_edges_1[~np.isinf(energy_edges_1)]
#
#    use_energies = np.where(energy_edges_all1 == energy_edges_1)[0]
#    print(f"use_energies {use_energies.size}")
#    #energy_edge_mask = np.array(range(33))
#    #energy_edge_mask[use_energies] = 1 #not actually used?
#
#    energy_edges_used = (energy_edges_all1 + energy_shift)[use_energies]
#    print(f"size energy_edges_used {energy_edges_used.size}")
#    out_mean, out_gmean, width, edges_1, edges_2 = edge_products(energy_edges_used)
#    energy_low = edges_2[:,0]
#    energy_high  = edges_2[:,1]
#    low_fsw_idx = use_energies[:-1] #used to work: use_energyes[1:]
#    high_fsw_idx = use_energies[1:]-1 #works for l1:use_energies[2:]-1
#    e_axis = stx_energy_axis(num_energy = len(use_energies) - 1, energy_mean = out_mean, gmean = out_gmean, width = width, low = energy_low, high = energy_high, low_fsw_idx = low_fsw_idx, high_fsw_idx = high_fsw_idx, edges_1 = edges_1, edges_2 = edges_2)
#
#    return Spectrogram(primary_header, spec_data, control, energy, t_axis, e_axis, fits_path, background = background)

def convert_spectrogram(fits_path_data, fits_path_bk = None, shift_duration = 0, energy_shift = 0, distance = 1.0,  flare_location= [0,0], elut_filename = None, replace_doubles = False, keep_short_bins = True, apply_time_shift = True, to_fits= False, use_discriminators = True):
    """Convert STIX spectrogram for use in XSPEC (translation of stx_convert_spectrogram.pro, which coverts STIX spectrograms for use with OPSEX) """
    spec = Spectrogram(fits_path_data, shift_duration = 1, replace_doubles = replace_doubles, keep_short_bins = keep_short_bins, background = False, use_discriminators = use_discriminators)
    dist_factor = 1./(spec.distance**2.)
   
    spec.initialize_spectrogram()

    if spec.counts.ndim == 2: #it's from a spectrogram
        spec.data_level = 4
        counts_spec = spec.counts
    else:
        counts_spec = np.sum(spec.counts,axis=0) #sum over detectors
    spec.correct_counts()
    
    print(".......... BACKGROUND ........")
    #background
    spec_bk = Spectrogram(fits_path_bk, shift_duration = None, replace_doubles = replace_doubles, keep_short_bins = keep_short_bins, background = True, use_discriminators = False)
    spec_bk.control_data.energy_bin_mask = spec.control_data.energy_bin_mask
    spec_bk.pixel_mask = spec.pixel_mask
    spec_bk.detector_mask = spec.detector_mask
    spec_bk.initialize_spectrogram(n_energies = spec.n_energies)
    spec_bk.n_energies = 32
    spec_bk.correct_counts()
    
    print(".......... BACKGROUND SUBTRACTION ........")
    #extra background corrections - stx_convert_science_data2ospex 153-190
    rdict = background_subtract(spec, spec_bk, counts_spec)
    
    if not to_fits: #for testing
        rdict['spec']=spec
        rdict['spec_bk']=spec_bk
        return rdict
    
    else: #write background-corrected counts to fits
        spec.counts = rdict['spec_in_corr']
        spec.error = rdict['total_error'].T
        spec.spectrum_to_fits(f"stx_spectrum_{spec.t_axis.time_mean[0] :%Y%m%d_%H%M%S}.fits")
#    #data_dims = np.array([n_energies, 1, 1, n_times])
#
#    # Get the rcr states and the times of rcr changes from the ql_lightcurves structure
#    #ut_rcr = spec.t_axis.time_start
#    #find_changes, rcr, index, state, count=count
#
##   ;add the rcr information to a specpar structure so it can be incuded in the spectrum FITS file
##   specpar = { sp_atten_state :  {time:ut_rcr[index], state:state} }
#

def bk_count_manipulations(bk_counts, duration, timedel, energy_bins, eff_ewidth, ntimes, name = 'corrected_counts_bk', error = False):
    # seems to operate differently on corrected_counts and spec_in_bk..
    if error:
        #dim1 = bk_counts.shape[-1]
        bk_counts = np.sqrt(np.sum(bk_counts**2,axis = 0)) #/np.sum(timedel)
        bk_counts = np.sqrt(bk_counts**2/np.sum(timedel))
        print_arr_stats(bk_counts)
    else:
        bk_counts = np.sum(bk_counts,axis = 0)/np.sum(timedel)
    
    bk_counts = bk_counts[energy_bins]

    if bk_counts.ndim == 1:
        bk_counts = np.expand_dims(bk_counts, 1)
    bk_counts = np.outer(duration, bk_counts.T)
    
    bk_counts =  bk_counts/np.tile(eff_ewidth, ntimes).reshape(( ntimes,energy_bins.size)) #[:,energy_bins]

    return bk_counts
    
def get_eff_livetime_fraction(spec):
    #corrected_counts = np.sum(corrected_counts.reshape((spec.n_energies, 1, spec.n_times)), axis=1) #not really necessary
    if spec.data_level == 4:
        eff_livetime_fraction = spec.counts_before_livetime/spec.counts
    else:
        eff_livetime_fraction = np.sum(spec.counts_before_livetime,axis=0)/spec.counts #,axis=0)
    eff_livetime_fraction[np.isnan(eff_livetime_fraction)] = 1
    #eff_livetime_fraction = np.mean(eff_livetime_fraction, axis = 0)
    if eff_livetime_fraction.shape != (spec.n_times, spec.n_energies): #(spec.n_energies, spec.n_times):
        eff_livetime_fraction_expanded = np.tile(eff_livetime_fraction,spec.n_energies).reshape((spec.n_energies,spec.n_times)) #unchanged for now
    else:
        eff_livetime_fraction_expanded = eff_livetime_fraction

    return eff_livetime_fraction_expanded
    
### energy stuff for later
#       emin = 1
#       emax = 150
#       new_edges = np.where(np.logical_and(self.e_axis.edges_1 > emin, self.e_axis.edges_1 <= emax))[0] #makes sense to be <= rather than just <
#
#       new_energy_edges = self.e_axis.edges_1[new_edges]
#       if not np.array_equal(self.e_axis.edges_1, new_energy_edges): #different shape or values
#           print("updating counts and errors")
#
#           out_mean, out_gmean, width, edges_1, edges_2 = edge_products(new_energy_edges)
#           energy_low = edges_2[:,0]
#           energy_high  = edges_2[:,1]
#           low_fsw_idx = new_edges[1:]
#           high_fsw_idx = new_edges[2:]-1
#
#           e_axis_new = stx_energy_axis(num_energy = len(new_energy_edges) - 1, energy_mean = out_mean, gmean = out_gmean, width = width, low = energy_low, high = energy_high, low_fsw_idx = low_fsw_idx, high_fsw_idx = high_fsw_idx, edges_1 = edges_1, edges_2 = edges_2)
#
#           #print(e_axis_new.__dict__)
#           new_energies = [i for i,e in enumerate(self.e_axis.energy_mean) if e in e_axis_new.energy_mean]
#           #print(self.e_axis.__dict__)
#           print(new_energies, len(new_energies), spec_in_corr.shape)
#           self.e_axis = e_axis_new
#           self.counts =  spec_in_corr[new_energies,:]
#           self.error = total_error[new_energies,:]
#       else:


def background_subtract(spectrogram, spectrogram_bk, counts_spec):
    corrected_counts = spectrogram.counts#.T
    corrected_error = spectrogram.error#.T

    corrected_error_bk = spectrogram_bk.error[...,0].T#[0]#np.sqrt(spectrogram_bk.error[0]**2)
    print("corrected_error_bk")
    print_arr_stats(corrected_error_bk)

    ntimes_bk, n_energies_bk, _, _ = spectrogram_bk.data['counts'].shape #i think it's time, energy,pixels, detectors ...
    #or time, detectors, pixels, detectors_used?
    detectors_used = np.where(spectrogram.detector_mask == 1)[0]
    pixels_used = np.where(spectrogram.pixel_mask == 1)[0]
    n_detectors_bk = detectors_used.size
    n_pixels_bk = pixels_used.size
    ntimes = spectrogram.n_times
    energy_bins = spectrogram.e_axis.low_fsw_idx +1 #, spectrogram.e_axis.high_fsw_idx[-1]+1)
    
    corrected_counts_bk = spectrogram_bk.counts[...,0].T#[:,0] #sum over detectors. shape goes from (30,32) to (32) #np.sum(spectrogram_bk.data['counts'][0][detectors_used][:,pixels_used,:],axis=1)#counts #should already be correct shape
    spec_in_bk = np.sum(spectrogram_bk.data['counts'][0][:,pixels_used,:][:,:,detectors_used],axis=1).T #sum over pixels
    
    print('original corrected counts and spec_in')
    print_arr_stats(corrected_counts_bk)
    print_arr_stats(spec_in_bk)
    #print(energy_bins)
    fig,ax=plt.subplots()
    ax.imshow(corrected_counts, origin = 'lower')
    ax.set_aspect('auto')
    ax.set_title("Corrected counts in ")
    fig.show()
    
    if spectrogram.t_axis.duration.ndim == 1:
        spectrogram.t_axis.duration = np.expand_dims(spectrogram.t_axis.duration,1)
    
    corrected_counts_bk = bk_count_manipulations(corrected_counts_bk, spectrogram.t_axis.duration, spectrogram_bk.data['timedel'], energy_bins, spectrogram.eff_ewidth, ntimes)
    print("corrected_counts_bk after")
    print_arr_stats(corrected_counts_bk)
    spec_in_bk = bk_count_manipulations(spec_in_bk, spectrogram.t_axis.duration, spectrogram_bk.data['timedel'], energy_bins, spectrogram.eff_ewidth, ntimes, name = 'spec_in_bk')
    
    error_bk = bk_count_manipulations(corrected_error_bk, spectrogram.t_axis.duration, spectrogram_bk.data['timedel'], energy_bins, spectrogram.eff_ewidth, ntimes, name = 'error_bk', error = True)
    print("error_bk after")
    print_arr_stats(error_bk)

    spec_in_corr = corrected_counts - corrected_counts_bk
    spec_in_uncorr = counts_spec - spec_in_bk #remove .T from counts_spec
    total_error = np.sqrt(corrected_error**2. + error_bk**2.)
    print(f"total_error nzero: {np.count_nonzero(total_error==0)}")
    
    print_arr_stats(spec_in_corr)
    eff_livetime_fraction_expanded = get_eff_livetime_fraction(spectrogram)
    spec_in_corr *= eff_livetime_fraction_expanded #.T
    total_error *= eff_livetime_fraction_expanded #.T
    
    #energy correction too if needed
    
    fig,ax=plt.subplots()
    ax.imshow(spec_in_corr, origin = 'lower')
    ax.set_aspect('auto')
    ax.set_title("Background Corrected counts")
    fig.show()
    
    rdict = {'spec_in_corr':spec_in_corr,'spec_in_uncorr':spec_in_uncorr,'corrected_counts':corrected_counts, 'corrected_counts_bk':corrected_counts_bk, 'counts_spec':counts_spec, 'spec_in_bk':spec_in_bk,'total_error':total_error,'corrected_error':corrected_error, 'error_bk':error_bk}
    return rdict

