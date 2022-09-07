import pandas as pd
import os
from datetime import datetime as dt

def science_energy_channels(basefile = "ScienceEnergyChannels_1000.csv", ql = False, _extra = False):
    """ This function returns the science energy bins. It reads the data file and holds the result in a dictionary"""
    df = pd.read_csv(f"{os.environ['STX_DET']}/{basefile}")
    
    if ql:
    
    return channels
