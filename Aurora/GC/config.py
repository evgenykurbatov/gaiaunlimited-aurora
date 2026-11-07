
import pathlib

import numpy as np


data_path = str(pathlib.Path.home() / 'IoA.GaiaUnlimited/_data/')

## GaiaUnlimited
gaiaunlimited_data_path = data_path + 'gaiaunlimited.data/'

## WSDB
wsdb_host = 'cappc127.ast.cam.ac.uk'
wsdb_user = 'evgeny_kurbatov'
#wsdb_password = 'yAvx4hEu@J'
wsdb_password = 'oEl;8xEazR'

## Magnitude bins
G_min,  G_max,  dG  = 1.7, 23.5, 0.2  ## mag
BP_min, BP_max, dBP = 2.0, 26.0, 0.2  ## mag
RP_min, RP_max, dRP = 1.0, 24.6, 0.2  ## mag

G_bins  = np.linspace(G_min,  G_max,  int((G_max-G_min)/dG) + 1)
BP_bins = np.linspace(BP_min, BP_max, int((BP_max-BP_min)/dBP) + 1)
RP_bins = np.linspace(RP_min, RP_max, int((RP_max-RP_min)/dRP) + 1)

#print("G_bins:", G_bins)
#print("shape(G_bins) =", G_bins.shape)
