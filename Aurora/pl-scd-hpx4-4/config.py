
import os
import sys
sys.path.append('..')

import numpy as np

import global_config


data_path = global_config.data_path


# Angular resolution
hpxorder = 4


# Cuts and limits
mh_lo_lim, mh_hi_lim = -1.3, -1.0
logg_cut = 3.0


# Density models
import phys_models
density_gse = phys_models.Density_Lane23_ELZ_SCDn()

# Model parameters
keys = ['b', 's', 'k', 'xi', 'lf1', 'lf2', 'lf3']


# Output options
_, save_suffix = os.path.split(os.getcwd())
