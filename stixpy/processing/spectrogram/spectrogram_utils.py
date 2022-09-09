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

def shift_one_timestep(arr_in, axis = 0, shift_step = -1):
    if shift_step == 0:
        return arr_in
    shifted_arr = np.copy(arr_in)
    shifted_arr = np.roll(arr_in, shift_step, axis = axis)
    if np.sign(shift_step) == -1:
        return shifted_arr[:shift_step]
    else:
        return shifted_arr[shift_step:]

def select_srm_channels(srm_edges, spec_edges):
    '''get channels of SRM to match channels of spectrum '''
    srm_list = srm_edges.tolist()
    keep_channels = [srm_list.index(e1.tolist()) for e1 in spec_edges if e1 in srm_edges]
    #if len(keep_channels) != len(srm_edges): #pad with zeros
    #    keep_channels.extend(np.zeros(len(srm_edges)-len(keep_channels)).tolist()) #this is the final mask
    return keep_channels
    
def write_cropped_srm(srm,keep_channels,fitsfilename=None):
    matrix_names = ['ENERG_LO','ENERG_HI','N_GRP','F_CHAN','N_CHAN','MATRIX']
    print(f"original shapes: {srm[1].data.MATRIX.shape}")
    matrix = srm[1].data
    new_nchan = np.ones(matrix.F_CHAN.size) + len(keep_channels)
    new_matrix = matrix.MATRIX[:,keep_channels] #can't do this have to make new ones
    
    matrix_table = Table([matrix.ENERG_LO, matrix.ENERG_HI, matrix.N_GRP, matrix.F_CHAN, new_nchan.astype('>i4'), new_matrix.astype('>f4')], names = matrix_names)
    
    ebounds = srm[2].data
    ebounds_names = ['CHANNEL','E_MIN','E_MAX']
    new_channel = ebounds.CHANNEL[keep_channels]
    #print(new_channel)
    new_emin = ebounds.E_MIN[keep_channels]
    new_emax = ebounds.E_MAX[keep_channels]

    ebounds_table = Table([new_channel.astype('>i4'), new_emin.astype('>f4'), new_emax.astype('>f4')], names = ebounds_names)

    #see how it goes without updating the headers manually
    matrix_HDU = fits.BinTableHDU(data = matrix_table, header = srm[1].header)
    ebounds_HDU = fits.BinTableHDU(data = ebounds_table, header = srm[2].header)
    hdul = fits.HDUList([srm[0], matrix_HDU, ebounds_HDU, srm[3]])
    print(f"final shapes: {srm[1].data.MATRIX.shape}")
    if not fitsfilename:
        fitsfilename = f"{srm[1].header['PHAFILE'][:-5]}_{len(keep_channels)}_chans.fits"
    hdul.writeto(fitsfilename)
    return fitsfilename

