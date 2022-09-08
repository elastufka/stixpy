import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy import constants
import warnings
from datetime import datetime as dt
from datetime import timedelta as td

def get_header_corrections(fits_path):
    """Returns distance from the Sun and time shift from primary header information"""
    primary_header =fits.open(fits_path)[0].header
    au = constants.au.value # Already in meters
    distance_sun_m = primary_header['DSUN_OBS']
    distance = distance_sun_m/au
    time_shift = primary_header['EAR_TDEL']
    return distance, time_shift
    
def open_spec_fits(filename):
    # Read the FITS file
    hdul = fits.open(filename)
    primary_header = hdul[0].header
    control = hdul[1]
    data = hdul[2]
    energy = hdul[3] if hdul[3].name == 'ENERGIES' else hdul[4]
    #data_header = data.header
    return primary_header, control, data, energy
    
def get_hstart_time(primary_header):
    try:
        hstart_str = primary_header['DATE_BEG']
    except KeyError:
        hstart_str = primary_header['DATE-BEG']
    hstart_time = dt.strptime(hstart_str,"%Y-%m-%dT%H:%M:%S.%f")
    return hstart_str, hstart_time
    
def get_use_detectors(det_ind = None):
    g10=np.array([3,20,22])-1
    g09=np.array([16,14,32])-1
    g08=np.array([21,26,4])-1
    g07=np.array([24,8,28])-1
    g06=np.array([15,27,31])-1
    g05=np.array([6,30,2])-1
    g04=np.array([25,5,23])-1
    g03=np.array([7,29,1])-1
    g02=np.array([12,19,17])-1
    g01=np.array([11,13,18])-1
    g01_10=np.concatenate([g01,g02,g03,g04,g05,g06,g07,g08,g09,g10]).tolist()
    g03_10=np.concatenate([g03,g04,g05,g06,g07,g08,g09,g10]).tolist()
    
    mask_use_detectors = np.zeros(32)
    if isinstance(det_ind, list):
        mask_use_detectors[det_ind] = 1
    else:
        mask_use_detectors[g03_10] = 1
    return mask_use_detectors
    
def get_use_pixels(pix_ind = None):
    if not pix_ind:
        return np.ones(12)
    elif isinstance(pix_ind, list):
        mask_use_pixels = np.zeros(12)
        mask_use_pixels[pix_ind] = 1
        return mask_use_pixels

def edge_products(edges):
    if edges.size == 1:
        return edges
    dims = edges.shape
    try:
        if dims[1] >= 2: # Already stacked... should be elow, ehigh
            edges_2 = edges
            edges_1 = edges[:,0]
            edges_1 = np.append(edges_1,edges[-1,1])
    except IndexError:
        n = dims[0]
        edges_2 = np.vstack([edges[:n-1], edges[1:]]).T
        edges_1 = edges
        
    out_mean = np.sum(edges_2, axis=1)/2.
    gmean = ((edges_2[:,0]*edges_2[:,1]))**0.5  # Geometric mean
    width = np.abs(edges_2[:,1] - edges_2[:,0]) # Width of bins
    if edges_2[0,0] == 0: #if first energy is 0
        width = width[1:]
    #width = width[~np.isinf(width)] #if there's an inf get rid of it
    return out_mean, gmean, width, edges_1, edges_2

def ogip_time_calcs(spec):
    #TIMESYS = '1979-01-01T00:00:00' / Reference time in YYYY MM DD hh:mm:ss
    #TIMEUNIT= 'd       '           / Unit for TIMEZERO, TSTARTI and TSTOPI

    #timezero = Time(spec.primary_header['DATE_BEG'], format = 'isot').mjd # zero time used to calculate the n-time event or the n-time
    #tstart = timezero
    #tstop = Time(spec.primary_header['DATE_END'], format = 'isot').mjd
    #mjdref = Time("1970-01-01T00:00:00.000000",format = "isot").mjd # MJD for reference time 01-01-1970
    #tmjd = Time((spec.data['time']+timezero*86400)/86400, format = 'mjd') #.isot to be readable
    
    #use datetimes in t_axis...(will this write to FITS correctly?)
    timecen = np.array(spec.t_axis.time_mean)
    
    factor = 1
    #units for L1 are centiseconds
    if spec.primary_header['LEVEL'].strip() == 'L1':
        factor = 100
    
    #calculate time parameters to be passed into the fits file
    specnum = np.arange(len(timecen)) +1
    channel = np.tile(np.arange(spec.n_energies), spec.n_times).reshape(spec.n_times,spec.n_energies) # array of n_times x n_channels
    #timecen =  tmjd + spec.data['timedel']/2.0 #doesn't actually do anything
    eff_livetime_fraction = spec._get_eff_livetime_fraction(expanded = False)
    exposure = np.sum((spec.data['timedel']/factor)*eff_livetime_fraction)
    
    return {"specnum": specnum, "channel": channel, "timedel": spec.data['timedel']/factor, "timecen": timecen, "exposure": exposure}

def shift_one_timestep(arr_in, axis = 0, shift_step = -1):
    if shift_step == 0:
        return arr_in
    shifted_arr = np.copy(arr_in)
    shifted_arr = np.roll(arr_in, shift_step, axis = axis)
    if np.sign(shift_step) == -1:
        return shifted_arr[:shift_step]
    else:
        return shifted_arr[shift_step:]

