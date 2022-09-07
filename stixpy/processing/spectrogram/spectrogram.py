import pandas as pd
import numpy as np
import os
from datetime import datetime as dt
from .read_elut import *
from .spectrogram_utils import *
from .livetime import *
from astropy.table import Table
from .spectrogram_axes import stx_energy_axis, stx_time_axis
from matplotlib import pyplot as plt
from ml_utils import print_arr_stats

class Spectrogram:
    def __init__(self, filename, background = False, energy_shift = 0, use_discriminators = True, replace_doubles = False, keep_short_bins = True, shift_duration = None, alpha = None, time_bin_filename = None, det_ind = None, pix_ind = None):
        self.filename = filename
        self.background = background
        self.det_ind = det_ind
        self.pix_ind = pix_ind
        self._from_fits(energy_shift = energy_shift, use_discriminators = use_discriminators, replace_doubles = replace_doubles, keep_short_bins = keep_short_bins, shift_duration = shift_duration, alpha = alpha, time_bin_filename = time_bin_filename)
        
    def _alpha_from_header(self, primary_header):
        processing_level = primary_header['LEVEL']
        alpha = 1 if processing_level.strip() == 'L1A' else 0
        self.alpha = alpha
        
    def _remove_short_bins(self, hstart_str, replace_doubles = False):
        counts_for_time_bin = sum(self.counts[1:10,:],1) # shape is ex. (32,)
        idx_short = np.where(counts_for_time_bin >= 1400)[0]

        # List when we have 1 second time bins, short or normal bins
        mask_long_bins = np.ones(self.n_time-1)
        
        min_time_table = pd.read_csv(f"{os.environ['STX_CONF']}/detector/min_time_index.csv")

        min_time = df.where(hstart_str < df.where(hstart_str > df[' start_date'])[' end_date']).dropna(how='all')['Mininmum time [cs]'].values[0]/10.
        print(f"min_time {min_time}")

        if idx_short.size > 0:
            idx_double = np.where(self.duration[idx_short] == min_time)[0]
            if idx_double.size > 0:
                idx_short_plus = [idx_short, idx_short[idx_double]-1] #check
            else:
                idx_short_plus = idx_short

        if idx_double.size > 0 and replace_doubles:
            mask_long_bins[idx_short] = 0
            self.duration[idx_short[idx_double]-1] = (self.duration[max(np.where(self.duration[0:idx_short[idx_double]] > 1))] + self.duration[min(np.where(self.duration[idx_short[idx_double]:-1] > 1)) + idx_short[idx_double]])/2.
        else:
            mask_long_bins[idx_short_plus] = 0

        idx_long = np.where(mask_long_bins == 1)[0]

        self.time_bin_center = self.time_bin_center[idx_long]
        self.n_time = self.time_bin_center.size
        self.duration = self.duration[idx_long]
        self.counts = self.counts[idx_long,:]
        self.counts_err = self.counts_err[idx_long,:]
        self.triggers = self.triggers[idx_long]
        self.triggers_err = self.triggers_err[idx_long]
        
    def _get_energy_edges(self, energy, energies_used, energy_shift):
        #nenergies = energies_used.size
        #print(f"energy bin mask {control.data.energy_bin_mask}")
        print(f"energies_used {energies_used}")
        energy_edges_2 = np.transpose([[energy.data.e_low[energies_used]], [energy.data.e_high[energies_used]]])
        _, _, _, energy_edges_1, _ = edge_products(energy_edges_2.squeeze())
        print(f"energy_edges_1 {energy_edges_1.size}")
        
        energy_edges_all2 = np.transpose([[energy.data.e_low], [energy.data.e_high]])
        _, _, _, energy_edges_all1, _ = edge_products(energy_edges_2.squeeze())
        print(f"energy_edges_all1 {energy_edges_all1.size}")

        #remove infs
        #energy_edges_all1 = energy_edges_all1[~np.isinf(energy_edges_all1)]
        #energy_edges_1 = energy_edges_1[~np.isinf(energy_edges_1)]
        
        use_energies = np.where(energy_edges_all1 == energy_edges_1)[0]
        print(f"use_energies {use_energies.size}")
        #energy_edge_mask = np.array(range(33))
        #energy_edge_mask[use_energies] = 1 #not actually used?

        energy_edges_used = (energy_edges_all1 + energy_shift)[use_energies]
        print(f"size energy_edges_used {energy_edges_used.size}")
        out_mean, out_gmean, width, edges_1, edges_2 = edge_products(energy_edges_used)
        return use_energies, out_mean, out_gmean, width, edges_1, edges_2
        
    def _get_used_indices(self):
        if self.alpha: #pixel data
            mask_use_detectors = get_use_detectors(self.det_ind)
            mask_use_pixels = get_use_pixels(self.pix_ind)
            
            pixels_used = np.where(np.logical_and(np.sum(self.pixel_mask[0],0), mask_use_pixels))[0]
            detectors_used = np.where(np.logical_and(self.detector_mask[0], mask_use_detectors))[0]
        
#        if self.n_times == 1: #background
#            pm_sum = np.sum(self.data['pixel_masks'],2)
#            pixels_used = np.where(np.logical_and(pm_sum[pm_sum > 0], mask_use_detectors == 1))[0] #no longer correct? #where(total(data_str.pixel_masks,2) gt 0 and mask_use_pixels eq 1)
#            detectors_used = np.where(np.logical_and(self.data['detector_masks'] == 1, mask_use_detectors == 1))
#
        else: #L4
            pixels_used = np.where(self.pixel_mask[0,:] == 1)[0]
            detectors_used = np.where(self.detector_mask[0,:] == 1)[0]
        return pixels_used, detectors_used
        
    def _get_eff_ewidth(self, pixels_used, detectors_used):
        energy_edges_used = np.append(self.e_axis.low_fsw_idx,self.e_axis.high_fsw_idx[-1]+1)
        gain, offset, adc, ekev_actual = read_elut(elut_filename = self.elut_filename)
        self.ekev_actual = ekev_actual #for testing
        
        ave_edge = np.mean(np.mean(np.swapaxes(ekev_actual, 0,2)[energy_edges_used][:,pixels_used][:,:,detectors_used],axis=1),axis=1)
    #   ave_edge  = mean(reform(ekev_actual[energy_edges_used-1, pixels_used, detectors_used, 0 ], n_energy_edges, n_pixels, n_detectors), dim = 2)
        _, _, ewidth, _, _ = edge_products(ave_edge)
        true_ewidth = self.e_axis.width[~np.isinf(self.e_axis.width)]
        
        eff_ewidth =  true_ewidth/ewidth
        self.eff_ewidth = eff_ewidth #for testing
        
    def _from_fits(self, energy_shift = 0, use_discriminators = True, replace_doubles = False, keep_short_bins = True, shift_duration = None, alpha = None, time_bin_filename = None):
        """Read spectrogram FITS file. Same function as stx_read_spectrogram_fits_file and stx_read_pixel_data_fits_file
        
        args:
        
        fits_path_data : str
        The path to the sci-xray-spec (or sci-spectrogram) observation file
        
        kwargs:
        
        background : bool, default = False
        Is the input file a background file or not
        
        energy_shift : optional, float, default=0.
        Shift all energies by this value in keV. Rarely needed only for cases where there is a significant shift in calibration before a new ELUT can be uploaded.
        
        alpha : bool, default=0
        Set if input file is an alpha e.g. L1A
        
        use_discriminators : bool, default=False
        an output float value

        shift_duration : None
        Shift all time bins by 1 to account for FSW time input discrepancy prior to 09-Dec-2021. N.B. WILL ONLY WORK WITH FULL TIME RESOLUTION DATA WHICH IS USUALLY NOT THE CASE FOR SPECTROGRAM DATA.
         """
        short_bins_dt = dt.strptime('2020-11-25T00:00:00',"%Y-%m-%dT%H:%M:%S")
        shift_duration_dt = dt.strptime('2021-12-09T00:00:00',"%Y-%m-%dT%H:%M:%S")
        energy_shift_low_dt = dt.strptime('2020-11-15T00:00:00',"%Y-%m-%dT%H:%M:%S")
        energy_shift_high_dt = dt.strptime('2021-12-04T00:00:00',"%Y-%m-%dT%H:%M:%S")

        distance, time_shift = get_header_corrections(self.filename)
        primary_header, control, data, energy = open_spec_fits(self.filename)
        data.data.counts_err = np.sqrt(data.data.counts_err**2 + data.data.counts)
        data.data.triggers_err = np.sqrt(data.data.triggers_err**2 + data.data.triggers)
        
        self.n_time = data.data['time'].size
        energies_used = np.where(control.data.energy_bin_mask == 1)[1]
        #nenergies = energies_used.size

        hstart_str, hstart_time = get_hstart_time(primary_header)
        if not alpha:
            self._alpha_from_header(primary_header)

        #trigger_zero should always be 0 as far as I know... it gets modified by mreadfits 2.26
    #    try:
    #        trigger_zero = data.header['TZERO3']
    #    except KeyError:
    #        trigger_zero = 0
    #    new_triggers = np.array(trigger_zero) + data.data.triggers # 1D array
    #    data.data.triggers[:] = new_triggers
    ##    except KeyError:
    ##        pass

        if hstart_time < shift_duration_dt:
            shift_duration = 1

        #If time range of observation is during Nov 2020 RSCW apply average energy shift by default
        if hstart_time > energy_shift_low_dt and hstart_time < energy_shift_high_dt:
            energy_shift = 1.6
            warnings.warn(f"Warning: Due to the energy calibration in the selected observation time a shift of -{energy_shift} keV has been applied to all science energy bins.")

        if not keep_short_bins and hstart_time < short_bins_dt:
            warnings.warn(f"Automatic short bin removal should not be attempted on observations before {short_bins_dt:%Y-%m-%d}")

        shift_step = 0
        if shift_duration is not None and hstart_time > shift_duration_dt:
            warnings.warn(f"Shift of duration with respect to time bins is no longer needed after {shift_duration_dt:%Y-%m-%d}")

        if shift_duration is not None and hstart_time < shift_duration_dt: # Shift counts and triggers by one time step - for use in background file?
            shift_step = -1 #otherwise default is zero and nothing happens
          
        print('shift step', shift_step)
        axis = 0 if self.alpha else -1 #time axis is last for pixel data...need to test this
        
        self.counts = shift_one_timestep(data.data.counts, shift_step = shift_step, axis = axis)
        self.counts_err = shift_one_timestep(data.data.counts_err, shift_step = shift_step, axis = axis)
        self.triggers = shift_one_timestep(data.data.triggers, shift_step = shift_step, axis = axis)
        self.triggers_err = shift_one_timestep(data.data.triggers_err, shift_step = shift_step, axis = axis)
        self.duration = shift_one_timestep(data.data.timedel, shift_step = -1*shift_step)
        self.time_bin_center = shift_one_timestep(data.data.time, shift_step = -1*shift_step)
        self.control_index = shift_one_timestep(data.data.control_index, shift_step = -1*shift_step)
        print('counts')
        print_arr_stats(self.counts)

        if not keep_short_bins:
            # Remove short time bins with low counts
            self._remove_short_bins(hstart_str, replace_doubles = replace_doubles)

        rcr = data.data.rcr # byte array
        if self.alpha: # things specific to L1A files
            try:
                rcr = control.data.rcr #need to reshape?
            except AttributeError:
                pass

        else:
            full_counts = np.zeros((self.n_time,32))
            full_counts[:, energies_used] = self.counts
            self.counts = full_counts.copy()

            full_counts_err = np.zeros((self.n_time, 32))
            full_counts[:,energies_used] = self.counts_err
            self.counts_err = full_counts_err
        
#        spec_data = {'time': time_bin_center,
#          'timedel': duration,
#          'triggers': triggers,
#          'triggers_err': triggers_err,
#          'counts': counts,
#          'counts_err': counts_err,
#          'control_index': control_index,
#          'rcr': rcr,
#          'header': data_header}
          
        if 'pixel_masks' in control.data.names:
            self.pixel_mask = control.data.pixel_masks
            self.detector_mask = control.data.detector_masks
        elif 'pixel_mask' in control.data.names:
            self.pixel_mask = control.data.pixel_mask
            self.detector_mask = control.data.detector_mask
        else:
            self.pixel_mask = data.data.pixel_masks
            self.detector_mask = data.data.detector_masks

        if self.background:
            self.pixel_mask = np.ones((1,12)) #will be changed in convert_spectrogram later
            self.detector_mask = np.ones((1,32))
            self.num_pixel_sets = data.data.num_pixel_sets
            self.num_energy_groups = data.data.num_energy_groups

        # Create time axis
        #TUNIT1 is Time unit
        if data.header['TUNIT1'].strip() == 's':
            factor = 1
        elif data.header['TUNIT1'].strip() == 'cs':
            factor = 100

        start_time = hstart_time + td(seconds = time_shift)
        t_start = [start_time + td(seconds = bc/factor - d/(2.*factor)) for bc,d in zip(self.time_bin_center, self.duration)]
        t_end = [start_time + td(seconds = bc/factor + d/(2.*factor)) for bc,d in zip(self.time_bin_center, self.duration)]
        t_mean = [start_time + td(seconds = bc/factor) for bc in self.time_bin_center]
        t_axis = stx_time_axis(time_mean = t_mean, time_start = t_start, time_end = t_end, duration = self.duration/factor)

        if (control.data.energy_bin_mask[0][0] or control.data.energy_bin_mask[0][-1]) and use_discriminators:
            control.data.energy_bin_mask[0][0] = 0
            control.data.energy_bin_mask[0][-1] = 0
            print("before use discriminators")
            print_arr_stats(self.counts)
            self.counts[...,0] = 0. #originally [0,:]
            self.counts[...,-1] = 0.
            self.counts_err[...,0] = 0.
            self.counts_err[...,-1] = 0.
            print_arr_stats(self.counts)
            
        energies_used = np.where(control.data.energy_bin_mask == 1)[1]
        use_energies, out_mean, out_gmean, width, edges_1, edges_2 = self._get_energy_edges(energy, energies_used, energy_shift)
        
        energy_low = edges_2[:,0]
        energy_high  = edges_2[:,1]
        low_fsw_idx = use_energies[:-1] #used to work: use_energyes[1:]
        high_fsw_idx = use_energies[1:]-1 #works for l1:use_energies[2:]-1
        e_axis = stx_energy_axis(num_energy = len(use_energies) - 1, energy_mean = out_mean, gmean = out_gmean, width = width, low = energy_low, high = energy_high, low_fsw_idx = low_fsw_idx, high_fsw_idx = high_fsw_idx, edges_1 = edges_1, edges_2 = edges_2)
        
        #probably don't need to keep original FITS stuff around but just in case for now
        self.primary_header = primary_header
        self.data_header = data.header
        self.data = data.data
        self.control_header = control.header
        self.control_data = control.data
        self.energy_header = energy.header
        self.energy_data = energy.data
        self.t_axis = t_axis
        self.e_axis = e_axis
        self.distance = distance
        self.hstart_str = hstart_str
        
    def initialize_spectrogram(self, elut_filename = None, n_energies = None):
        """All the stuff that happens after stx_read_..._fits_file and before stx_convert_science_data2ospex. needs a better name"""
        self.data_level = [int(c) for c in self.primary_header['LEVEL'].strip() if c in ['1','4']][0] # for now
            
        # Find corresponding ELUT
        if not elut_filename:
            self.elut_filename = date2elut_file(self.hstart_str)
            
        counts_in = self.counts
        dim_counts = counts_in.shape
        self.n_times = 1

        if len(dim_counts) > 1:
            self.n_times = dim_counts[0] #correct for both L1 and L4

        energy_bin_mask = self.control_data.energy_bin_mask
        energy_bins = np.where(energy_bin_mask[0] == 1)[0]
        if not n_energies:
            self.n_energies = len(energy_bins)
        else:
            self.n_energies = n_energies

        pixels_used, detectors_used = self._get_used_indices()
        
        pixel_mask_used = np.zeros(12)
        pixel_mask_used[pixels_used] = 1
        self.n_pixels = int(sum(pixel_mask_used))

        detector_mask_used = np.zeros(32)
        detector_mask_used[detectors_used] = 1
        self.n_detectors = int(sum(detector_mask_used))

        if not self.background:
            self._get_eff_ewidth(pixels_used, detectors_used)
        
            if not self.alpha: #L4
                spec_in = counts_in.T.copy()
                counts_spec =  spec_in[energy_bins,:] / np.repeat(self.eff_ewidth, self.n_times).reshape((self.n_energies, self.n_times))
                print('counts_spec l4')
                print_arr_stats(counts_spec)
                counts_err = self.data['counts_err'][:,energy_bins] #probably wrong... need a sqrt([]**2)?
            else: #L1
                counts_spec = np.moveaxis(self.counts,0,3) #[n_energies, n_pixels,n_detectors,n_times]
#                spec_in = np.sum(counts_in[:,pixels_used,:,:], 1) #sum over pixels, shape is 32, 32,n_times
                counts_spec = counts_spec[energy_bins]/np.reshape(np.tile(self.eff_ewidth, self.n_times*32*12),(self.n_energies,12,32,self.n_times))
#
                counts_err = np.moveaxis(self.counts_err,0,3) #[32,n_pixels,n_detectors,n_times]
                #counts_err = np.sqrt(np.sum(counts_err[:,pixels_used,:,:]**2,1))
                counts_err = counts_err[energy_bins]/np.reshape(np.tile(self.eff_ewidth, self.n_times*32*12),(self.n_energies,12,32,self.n_times))
        else: #still move time axis to the last position to make things easier
            counts_spec = np.moveaxis(self.counts,0,3) #[energy_bins] #self.data['counts']
            counts_err  = np.moveaxis(self.counts_err,0,3) #[energy_bins] #counts_err = self.data['counts_err']
                            
        if self.alpha: #L1 and background, sum over pixels
            print('counts_spec L1')
            print_arr_stats(counts_spec)
            print(f"detectors used: {detectors_used}\npixels_used: {pixels_used}")

            counts_spec = np.sum(counts_spec[:,pixels_used][:,:,detectors_used], axis = 1)
            counts_err = np.sqrt(np.sum(counts_err[:,pixels_used][:,:,detectors_used]**2, axis = 1))
            print('counts_spec after selection')
            print_arr_stats(counts_spec)

        #insert the information from the telemetry file into the expected stx_fsw_sd_spectrogram structure
        self.type = "stx_spectrogram"
        self.counts = counts_spec

        if len(self.data['triggers'].shape) == 1:
            self.trigger = self.data['triggers'].reshape((1,len(self.data['triggers'])))
            self.trigger_err = self.data['triggers_err'].reshape((1,len(self.data['triggers_err'])))

        elif not self.background:
            self.trigger = self.data['triggers'].squeeze().T
            self.trigger_err = self.data['triggers_err'].squeeze().T
        else:
            self.trigger = self.data['triggers'].T
            self.trigger_err = self.data['triggers_err'].T
        print('trigger shape',self.trigger.shape)
        self.pixel_mask = pixel_mask_used
        self.detector_mask = detector_mask_used
        self.rcr = self.data['rcr']
        self.error = counts_err
        
        
    def correct_counts(self):

#        if self.alpha:
#            #dim1 = self.counts.shape[1]
#
#            pixels_used = np.where(self.pixel_mask == 1)[0]
#            detectors_used = np.where(self.detector_mask== 1)[0]
#            n_detectors = self.detector_mask.sum()
#
#            #corrected_counts = self.counts[:,detectors_used,:]
#            #corrected_error = self.error[:,detectors_used,:]
#
#            print('corrected counts shape',corrected_counts.shape)
#            self.counts = corrected_counts
#            print('counts shape',self.counts.shape)
#            self.error = corrected_error
#
#        else:
#            dim1 = 1
        
        self.counts_before_livetime = self.counts.copy()
        corrected_counts, corrected_error, livetime_frac = spectrogram_livetime(self, level = self.data_level) #4?
        
        #if self.alpha:
        #    corrected_counts = np.sum(corrected_counts, axis=0) # Sum over detectors
        #    #livetime_frac = np.sum(livetime_frac, axis=1) # not technically necessary
       #     corrected_error = np.sqrt(np.sum(corrected_error**2,axis=0))#[:,0] # for now - is it possible that there are background files without ntimes = 1?
        
        print('corrected_counts')
        print_arr_stats(corrected_counts)
        print('corrected_error')
        print_arr_stats(corrected_error)
        self.counts = corrected_counts
        self.error = corrected_error
        self.livetime_fraction = livetime_frac
        #self.trigger = np.transpose(self.trigger)
        #print("self.counts")
        #print_arr_stats(self.counts)
        #print("self.error")
        #print_arr_stats(self.error)
        
    def select_energy_channels(self, elow):
        """Trim converted data to match the channels in an existing srm file, since unable to generate srm files via Python at the moment """
        chan_idx = [list(self.e_axis.low).index(e) for e in self.e_axis.low if e in elow]
        
        #fix energy axis
        self.e_axis.num_energy = len(chan_idx)
        for a in ['energy_mean','gmean','low','high','low_fsw_idx','high_fsw_idx','edges_1','edges_2']:
            attr = getattr(self.e_axis,a)
            setattr(self.e_axis,a,attr[chan_idx])
        
        #fix rate array etc
        self.n_energies = len(chan_idx)
        self.counts = self.counts[:,chan_idx]
        self.error = self.error[chan_idx,:]
    
    def spectrum_to_fits(self, fitsfilename, srm_file = "/Users/wheatley/Documents/Solar/STIX/demo_feb22/stx_srm_20210417_1531.fits"):
        reference_file = "/Users/wheatley/Documents/Solar/STIX/demo_feb22/stx_spectrum_20210417_1531.fits"
        srm = fits.open(srm_file) # Need to match the number of channels in here!
        self.select_energy_channels(srm[2].data.E_MIN)
        srm.close()
        
        reference_fits = fits.open(reference_file)#self.filename)
        timedict = ogip_time_calcs(self)
        # Make the primary header
        primary_header = reference_fits[0].header.copy()
        # Update keywords that need updating

        # Make the rate table
        rate_header = reference_fits[1].header.copy()
        # Update keywords that need updating
        rate_header['DETCHANS'] = self.n_energies
        rate_header['ONTIME'] = timedict['exposure']
        rate_header['EXPOSURE'] = timedict['exposure'] #should this be an int?
        #also update: timezero, tstarti, tstartf, tstopi, tstopf, telapse
        rate_names = ['RATE', 'STAT_ERR', 'CHANNEL', 'SPEC_NUM', 'LIVETIME', 'TIME', 'TIMEDEL']
        rate_table = Table([self.counts, self.error.T, timedict['channel'].astype('>i4'), timedict['specnum'].astype('>i2'), self.livetime_fraction, Time(timedict['timecen']).mjd, timedict['timedel'].astype('>f4')], names = rate_names) #is spec.counts what we want?

        # Make the energy channel table
        energy_header = reference_fits[2].header.copy()
        # Update keywords that need updating
        ct_edges_2 = self.e_axis.edges_2
        energy_names = ('CHANNEL', 'E_MIN', 'E_MAX')
        energy_table = Table([timedict['channel'][0].astype('>i4'), ct_edges_2[:,0].astype('>f4'), ct_edges_2[:,1].astype('>f4')], names = energy_names)

        # Make the attenuator state table
    #     att_header = reference_fits[3].header.copy()
    #     # Update keywords that need updating
    #     att_names = ('SP_ATTEN_STATE$$TIME', 'SP_ATTEN_STATE$$STATE')
    #     att_table = Table(spec[''], spec['e_axis'].e_min, names = att_names)
        reference_fits.close()

        primary_HDU = fits.PrimaryHDU(header = primary_header)
        rate_HDU = fits.BinTableHDU(header = rate_header, data = rate_table)
        energy_HDU = fits.BinTableHDU(header = energy_header, data = energy_table)
        print(energy_HDU.header)
        hdul = fits.HDUList([primary_HDU, rate_HDU, energy_HDU]) #, att_header, att_table])
        hdul.writeto(fitsfilename)
        
    
