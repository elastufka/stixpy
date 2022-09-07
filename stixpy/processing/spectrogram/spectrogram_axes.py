import pandas as pd
import numpy as np
import os
from datetime import datetime as dt

class stx_time_axis:
    def __init__(self, time_mean = None, time_start = None, time_end = None, duration = None):
        for k,v in locals().items():
            if k != 'self':
                setattr(self,k,v)
        self.type = 'stx_time_axis'
        
class stx_energy_axis:
    def __init__(self, num_energy = 32, energy_mean = None, gmean = None, low = None, high = None, low_fsw_idx = None, high_fsw_idx = None, edges_1 = None, edges_2 = None, width = None):
        for k,v in locals().items():
            if k != 'self':
                if v is not None:
                    setattr(self,k,v)
                else:
                    setattr(self,k,self.default_value(k))
        self.type = 'stx_energy_axis'

    def default_value(self, k):
        defaults = {'energy_mean': np.zeros(self.num_energy),
                    'gmean': np.zeros(self.num_energy),
                    'low': np.zeros(self.num_energy),
                    'high': np.zeros(self.num_energy),
                    'low_fsw_idx': list(range(self.num_energy)),
                    'high_fsw_idx': list(range(self.num_energy)),
                    'edges_1': np.zeros(self.num_energy + 1),
                    'edges_2': np.zeros((2,self.num_energy)),
                    'width': np.zeros(self.num_energy)
                   }
        return defaults[k]
