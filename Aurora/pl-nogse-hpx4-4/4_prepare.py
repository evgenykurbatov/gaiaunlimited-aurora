# %%

import sys
sys.path.append('..')
import pickle

import importlib
from tqdm import tqdm

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import pandas as pd

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord

import matplotlib as mpl
import matplotlib.pyplot as plt

from domain import Domain
import utils

import config


hpxorder = config.hpxorder

# %%
# Load everything

#
# Load domain

print("\n* Load domain")

dom = Domain().load(config.data_path + f"/domain_hpx{hpxorder}.npz")
D = dom.D
D_edges = dom.D_edges
G = dom.G
G_edges = dom.G_edges
dG = dom.dG
dVol = dom.dVol

#
# Load subsample

print("\n* Load subsample")

df0 = pd.read_hdf(config.data_path + '/arc23aurgb.hdf')
print("Num. of subsample stars:", len(df0))

# Cut by [M/H]
mh_lo_lim = config.mh_lo_lim
mh_hi_lim = config.mh_hi_lim
mask = (mh_lo_lim <= df0['mh_xgboost']) & (df0['mh_xgboost'] < mh_hi_lim)

# Cut by log(g)
mask &= (df0['logg_xgboost'] < config.logg_cut)

df0 = df0[mask].copy()
print(f"Num. of subsample stars after selection:", len(df0))

# Downscale healpixel indices if necessary
if hpxorder < 7:
    df0['hpx'] = df0['hpx'] // 4**(7-hpxorder)

# %%
# Prepare observables

print("\n* Prepare observables")

df0['d_bin'] = pd.cut(df0['dist'], bins=dom.D_edges, labels=False)
df0['g_bin'] = pd.cut(df0['phot_g_mean_mag'], bins=dom.G_edges, labels=False)

df = df0[['hpx', 'd_bin', 'g_bin']].copy()
print("NaNs:")
print(df.isna().sum())

df.dropna(inplace=True)
df = df.astype(int).copy()

cnt = df.groupby(['hpx', 'd_bin', 'g_bin'], observed=False).size().reset_index(name='cnt')

N_lbDG = np.zeros((dom.npix, len(dom.D), len(dom.G)))
N_lbDG[cnt['hpx'], cnt['d_bin'], cnt['g_bin']] = cnt['cnt']
print("shape(N_lbDG) =", N_lbDG.shape)

N_lbD = N_lbDG.sum(axis=-1)
N_lbG = N_lbDG.sum(axis=-2)
print("sums:", N_lbDG.sum(), N_lbD.sum(), N_lbG.sum())

# %%



# %%
# Save the observables

with open('observables.pkl', 'wb') as f:
    pickle.dump(N_lbDG, f)

# %%



# %%
# Diagnostics
#

print("sum(N_lbDG)  =", N_lbDG.sum())
print("sum(N_lbDG < 0)  =", (N_lbDG < 0).sum())
print("sum(N_lbDG == 0) =", (N_lbDG == 0).sum())

# %%
# Plot sky projection

plt.rc('font', size=6.0)
inch = 2.54  ## cm
width, height = 10/inch, 5/inch
plt.figure(figsize=(width, height), layout='constrained')
plt.suptitle(f"Num. of sources per healpix {hpxorder}")

plt.gca().set_aspect('equal')
sky = np.nan * np.zeros(dom.npix)
sky[dom.ipix] = N_lbG.sum(axis=1)
utils.plot_hpxmap(sky, vmin=1.0, interp=False, title="Aurora + GSE")

plt.savefig("fig/data_sky.pdf")
#plt.show()
plt.close()

# %%
# Plot luminosity functions

bs = [-30, -12, -5, 5, 20]
ls = [-60, -10, 0, 10, 20]
co = SkyCoord(ls, bs, frame='galactic', unit='deg')
hp = HEALPix(nside=dom.nside, order='nested', frame='icrs')
ipix_ = hp.skycoord_to_healpix(co)

plt.rc('font', size=6.0)
inch = 2.54  ## cm
width, height = 6/inch, 5/inch
plt.figure(figsize=(width, height), layout='constrained')
plt.suptitle(f"Luminosity function per healpix {hpxorder}")

for i, hpx in enumerate(ipix_):
    plt.stairs(N_lbG[hpx], G_edges, alpha=0.7, label=f"(l, b) = {ls[i], bs[i]}")
plt.legend()
plt.xlabel("G [mag]")
plt.yscale('log')
plt.ylim(ymin=0.3)

plt.savefig("fig/data_lumfun.pdf")
#plt.show()
plt.close()

# %%
