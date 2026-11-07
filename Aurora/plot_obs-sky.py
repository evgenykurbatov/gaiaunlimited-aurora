# %%

import importlib
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm

import global_config as config
import utils
importlib.reload(utils)


hpxorder = 4
hpxbase = 2**35 * 4**(12 - hpxorder)
nside = 2**hpxorder
npix = 12 * 4**hpxorder
print(f"hpxorder={hpxorder} -> (nside={nside}, npix={npix})")

# %%
# Read samples

#mh_edges = [-3.0, -2.0, -1.5, -1.3, -1.0]
mh_edges = [-3.0, -2.0, -1.3, -1.0, -0.5]

# Read catalogue
rgb0 = pd.read_hdf(config.data_path + '/arc23aurgb.hdf', columns=['source_id', 'mh_xgboost', 'logg_xgboost'])

obs = {}
for i in range(len(mh_edges)-1):
    mh_lo_lim, mh_hi_lim = mh_edges[i], mh_edges[i+1]
    key = "%.1f < [M/H] < %.1f" % (mh_lo_lim, mh_hi_lim)

    # Select stars by metallicity
    mask = (mh_lo_lim <= rgb0['mh_xgboost']) & (rgb0['mh_xgboost'] < mh_hi_lim)
    # ...and log(g)
    mask &= rgb0['logg_xgboost'] < config.logg_cut
    #
    rgb = rgb0[mask].copy()
    print(key, ": N =", len(rgb))

    # Count stars
    rgb['hpx'] = rgb['source_id'] // hpxbase
    #N_lb = np.nan * np.zeros(npix)
    N_lb = np.zeros(npix)
    tmp = rgb.groupby(['hpx'], observed=False).size().reset_index(name='cnt')
    N_lb[tmp['hpx']] = tmp['cnt']
    obs[key] = N_lb

# %%
# Plot

print("\n* Plot")

plt.rc('font', size=5.0)
inch = 2.54  ## cm
width, height = 17/inch, 9.5/inch
#plt.figure(figsize=(width, height), layout='constrained')
plt.figure(figsize=(width, height))

#norm = LogNorm(vmin=1e0, vmax=1e3)
norm = LogNorm(vmin=1, vmax=6000)
#norm = PowerNorm(vmin=0, vmax=1e3, gamma=0.1)
#norm = None

#cmap = 'inferno'
cmap = 'magma'
#cmap = 'Greys'
#cmap = 'binary'

keys = list(obs.keys())

plt.subplot(2, 2, 1)
plt.gca().set_aspect('equal')
key = keys[0]
N = obs[key]
print(f"{key}:  min, max =", np.nanmin(N), np.nanmax(N))
title = key
#title = key + f"; N = {int(np.nansum(N))}"
utils.plot_hpxmap(N, norm=norm, interp=False, cmap=cmap, xlabel=None, ylabel=None, cbar=False, title=title)

plt.subplot(2, 2, 2)
plt.gca().set_aspect('equal')
key = keys[1]
N = obs[key]
print(f"{key}:  min, max =", np.nanmin(N), np.nanmax(N))
title = key
#title = key + f"; N = {int(np.nansum(N))}"
utils.plot_hpxmap(N, norm=norm, interp=False, cmap=cmap, xlabel=None, cbar=False, title=title)

plt.subplot(2, 2, 3)
plt.gca().set_aspect('equal')
key = keys[2]
N = obs[key]
print(f"{key}:  min, max =", np.nanmin(N), np.nanmax(N))
title = key
#title = key + f"; N = {int(np.nansum(N))}"
utils.plot_hpxmap(N, norm=norm, interp=False, cmap=cmap, ylabel=None, cbar=False, title=title)

plt.subplot(2, 2, 4)
plt.gca().set_aspect('equal')
key = keys[3]
N = obs[key]
print(f"{key}:  min, max =", np.nanmin(N), np.nanmax(N))
title = key
#title = key + f"; N = {int(np.nansum(N))}"
im = utils.plot_hpxmap(N, norm=norm, interp=False, cmap=cmap, cbar=False, title=title)

#plt.subplots_adjust(right=0.82)
plt.subplots_adjust(left=0.075, right=0.9, top=0.95, bottom=0.1)
cbar_ax = plt.axes([0.925, 0.125, 0.025, 0.8])
plt.colorbar(im, cax=cbar_ax)

#plt.show()
plt.savefig("fig/data-obs-sky.pdf")
plt.close()
