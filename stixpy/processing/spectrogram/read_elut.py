import pandas as pd
import os
import numpy as np
from datetime import datetime as dt
import glob

def date2elut_file(date, stx_conf=os.environ['STX_CONF']):
    elut_index=glob.glob(f"{stx_conf}/elut/elut_index.csv")[0]
    elut_df=pd.read_csv(elut_index)
    elut_df[' end_date'].replace('none',dt.now().strftime("%Y-%m-%dT%H:%M:%S"),inplace=True)
    elut_df['start_date'] = pd.to_datetime(elut_df[' start_date'])
    elut_df['end_date'] = pd.to_datetime(elut_df[' end_date'])
    #elut_df.drop(columns=[' start_date',' end_date'], inplace=True)
    if isinstance(date,str):
        date = pd.to_datetime(date)
    elut_filename = elut_df.query("@date > start_date and @date <= end_date")[' elut_file'].iloc[0]# elut filename that applies to desired date
    return elut_filename

def read_elut(elut_filename = None, scale1024 = True, ekev_actual = True, stx_conf=os.environ['STX_CONF']):
    """ This function finds the most recent ELUT csv file, reads it, and returns the gain and offset used to make it along with the edges of the Edges in keV (Exact) and ADC 4096, rounded """
    if not elut_filename: # Try and get from cwd
        elut_file = glob.glob(f"{stx_conf}/elut/elut_table*.csv")
        
    elut = pd.read_csv(f"{stx_conf}/elut/{elut_filename}", header = 2)
        
    if scale1024:
        scl = 4.0
    else:
        scl = 1.0
        
    offset = (elut.Offset.values / scl).reshape((32,12))
    gain = (elut["Gain keV/ADC"].values * scl).reshape((32,12))
        
    adc4096 = np.transpose(elut.values[:,4:].T.reshape((31,32,12)), axes = (0,2,1)).T # 31 x 12 x 32 but in correct order
    science_energy_channels = pd.read_csv(f"{stx_conf}/detector/ScienceEnergyChannels_1000.csv", header = 21, skiprows = [22,23])
    ekev = pd.to_numeric(science_energy_channels['Energy Edge '][1:32]).values
    adc4096_dict = {"ELUT_FILE": elut_filename,
                "EKEV": ekev, # Science energy channel edges in keV
                "ADC4096": adc4096, # 4096 ADC channel value based on EKEV and gain/offset
                "PIX_ID": elut.Pixel.values.reshape((32,12)).T, # Pixel cell of detector, 0-11
                "DET_ID": elut.Detector.values.reshape((32,12)).T, # Detector ID 0-31
                }
    if ekev_actual:
        gain4096 = np.zeros((32,12,31))
        offset4096 = np.zeros((32,12,31))
        for i in range(31):
            gain4096[:,:,i] = gain/scl
            offset4096[:,:,i] = offset * scl
        ekev_act = (adc4096 - offset4096) * gain4096
        return gain, offset, adc4096_dict, ekev_act
    else:
        return gain, offset, adc4096_dict
