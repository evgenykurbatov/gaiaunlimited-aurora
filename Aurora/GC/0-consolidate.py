# %%

import sys
import pathlib
import re
from tqdm import tqdm

import numpy as np
import scipy as sp
import pandas as pd

import astropy.units as u
from astropy.table import Table

import matplotlib as mpl
import matplotlib.pyplot as plt

import config

# %%
# Load Baumgardt catalogue of the Structural Parameters of Galactic Globular Clusters
# https://people.smp.uq.edu.au/HolgerBaumgardt/globular/parameter.html
#
# ASCII file with the combined data of all clusters:
# https://people.smp.uq.edu.au/HolgerBaumgardt/globular/combined_table.txt

colnames = ['name', 'ra', 'dec', 'D', 'sigma_D', 'D_GC', 'sigma_D_GC', 'N_RV', 'N_PM', 'Mass', 'sigma_Mass',
            'V', 'sigma_V', 'M/L_V', 'sigma_(M_L_V)', 'r_c', 'r_hl', 'r_hm', 'r_t',
            'rho_c', 'rho_hm', 'T_rh', 'Mass_ini', 'T_Diss', 'Mass_low', 'Mass_high',
            'alpha', 'sigma_0', 'v_esc', 'eta_c', 'eta_h', 'A_rot', 'P_rot']
baum = pd.read_table('Baumgardt_combined_table.txt', skiprows=3, delimiter='\s+', header=None,
                     names=colnames,
                     usecols=np.arange(len(colnames)), dtype={'name': str})

print("baum.columns =", baum.columns.tolist())
print("NaNs:", baum.isna().sum().sum())

baum

# %%
# Load Belokurov catalogue of the GC parameters (private communication)

bel = Table.read('gc_catalog_full.fits', format='fits').to_pandas()

# Convert column names to lower case
bel.columns = [ c.lower()  for c in bel.columns.tolist() ]
print("bel.columns =", bel.columns.tolist())
print("NaNs:", bel.isna().sum().sum())

# Bytes to string
bel['name']    = bel.apply(lambda r: r['name'].decode('utf-8').strip().replace(' ', '_'), axis=1)
bel['vaname']  = bel.apply(lambda r: r['vaname'].decode('utf-8').strip().replace(' ', '_'), axis=1)
bel['prog_ma'] = bel.apply(lambda r: r['prog_ma'].decode('utf-8').strip(), axis=1)

bel

# %%
# Rename some labels in the Baumgardt catalogue to harmonize them with Belokurov catalogue

# 'Ter_*' -> 'Terzan_*'
baum['name'] = baum.apply(lambda r: re.sub("^Ter", "Terzan", r['name']), axis=1).tolist()
# 'Djor_*' -> 'Djorg_*'
baum['name'] = baum.apply(lambda r: re.sub("^Djor", "Djorg", r['name']), axis=1).tolist()

# %%
# Rename awkward columns in the Baumgardt catalogue

columns_map = { 'M/L_V': 'M_to_L_V', 'sigma_(M_L_V)': 'sigma_M_to_L_V' }
baum.rename(columns=columns_map, inplace=True)

# %%
# Merge the Baumgardt and Belokurov catalogues

gc = baum.merge(bel, on='name', how='inner', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
print("len(gc) =", len(gc))
print("gc.columns =", gc.columns.tolist())
print("NaNs:", gc.isna().sum().sum())

# %%
# Save the Baumgardt + Belokurov catalogue

# Write out the columns list
with open('baum+bel-gc-columns.txt', 'wt') as f:
    [ f.write(f"{c}\n")  for c in gc.columns ]

print(gc.columns.tolist())
gc.to_hdf('baum+bel-gc.hdf', key='gc', format='table', index=False, data_columns=True, mode='w')

# %%




# %%
# Load and consolidate catalogues of GC stars by Vasiliev & Baumgardt
# https://zenodo.org/records/4891252

data_path = config.data_path + '/Vasiliev21/'

gcstars_ = []
for i in range(len(gc)):
    name, vaname = gc.iloc[i]['name'], gc.iloc[i]['vaname']
    fnames = [ f.name  for f in pathlib.Path(data_path + '/catalogues').glob(f"{vaname}.txt") ]
    print(f"{name} | {vaname} ->", fnames)
    if fnames == []:
        continue
    fname = fnames[0]

    # Read stars
    tmp = pd.read_csv(data_path + 'catalogues/' + fname, delimiter='\s+', comment='#',
                      names=['source_id', 'ra', 'dec', 'x', 'y', 'plx', 'pmra', 'pmdec', 'plxe', 'pmrae', 'pmdece', 'pmcorr',
                             'g_mag', 'bp_rp', 'Sigma', 'qflag', 'memberprob'],
                      dtype={'source_id': int, 'qflag': int})
    tmp['vaname'] = vaname
    gcstars_.append(tmp)

gcstars = pd.concat(gcstars_, ignore_index=True)
gcstars

# %%
# Save Vasiliev & Baumgardt consolidated catalogue

# Write out the columns list
with open('vasiliev2021-gcstars-columns.txt', 'wt') as f:
    [ f.write(f"{c}\n")  for c in gcstars.columns ]

print(gcstars.columns.tolist())
gcstars.to_hdf('vasiliev2021-gcstars.hdf', key='gcstars', format='table', index=False, data_columns=True, mode='w')

# %%
