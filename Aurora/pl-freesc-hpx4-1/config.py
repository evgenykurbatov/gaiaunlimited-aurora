
import os
import sys
sys.path.append('..')

import numpy as np

import global_config


data_path = global_config.data_path


# Angular resolution
hpxorder = 4


# Cuts and limits
mh_lo_lim, mh_hi_lim = -3.0, -2.0
logg_cut = 3.0


# Model parameters
keys = ['b1', 's1', 'k1', 'a2', 'b2', 'c2', 'k2', 'r2', 'theta', 'eta', 'phi', 'xi', 'lf1', 'lf2', 'lf3']


# Output options
_, save_suffix = os.path.split(os.getcwd())
#print(save_suffix)
