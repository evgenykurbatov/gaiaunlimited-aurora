#%%

import numpy as np
import scipy as sp
import pandas as pd

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, ICRS, Galactic, Galactocentric
import dustmaps
import dustmaps.config
import dustmaps.gaia_tge

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import config
import utils


# Use HEALPix order 7
hpxorder = 7
hpxbase = 2**35 * 4**(12 - hpxorder)
nside = 2**hpxorder
npix = 12 * 4**hpxorder
print(f"hpxorder={hpxorder} -> (nside={nside}, npix={npix})")

#%%
# Load Andrae et al. (2023) RGB catalog
#

print("\n* Load Andrae et al. (2023) RGB catalog")

rgb = pd.read_hdf(config.data_path + '/arc23rgb.hdf')
print("  records:", len(rgb))
print("  memory: %.1f MiB" % (rgb.memory_usage().sum()/(1024**2)))

# %%

print(rgb[['teff_xgboost', 'logg_xgboost', 'mh_xgboost', 'a0', 'ag', 'abp', 'arp']].describe())

#%%
# Load dustmap
# Manual: https://dustmaps.readthedocs.io/en/latest/
#

print("\n* Load dustmap")

dustmaps.config.config['data_dir'] = config.data_path + '/dustmaps.data'
dustmaps.gaia_tge.fetch()
dm = dustmaps.gaia_tge.GaiaTGEQuery(healpix_level=hpxorder)

#%%
#
# Get A0 map
#

print("\n* Get A0 map")

hp = HEALPix(nside=nside, order='nested', frame=ICRS())
co = hp.healpix_to_skycoord(np.arange(npix))

A0_map = dm.query(co)
print("min(A0), max(A0) =", np.nanmin(A0_map), np.nanmax(A0_map))

# Substitute NaNs by 20, i.e. holow pixels are all-extincting pixels
A0_map[np.isnan(A0_map)] = 20.0

del(dm)

#%%
#
# Fit extinctions
#

print("\n* Get extinction maps")

print("rgb.types:")
print(rgb.dtypes)
print()

def Am_fit(A0, c1, c2, c3, c4):
    return c1*A0 + c2*A0**2 + c3*A0**3 + c4*A0**4

# G
popt, _ = sp.optimize.curve_fit(Am_fit, rgb['a0'], rgb['ag'])
AG_fit_coef = list(popt)
AG_map = Am_fit(A0_map, *AG_fit_coef)
print("AG_fit_coef: ", AG_fit_coef)

# BP
popt, _ = sp.optimize.curve_fit(Am_fit, rgb['a0'], rgb['abp'])
ABP_fit_coef = list(popt)
ABP_map = Am_fit(A0_map, *ABP_fit_coef)
print("ABP_fit_coef:", ABP_fit_coef)

# RP
popt, _ = sp.optimize.curve_fit(Am_fit, rgb['a0'], rgb['arp'])
ARP_fit_coef = list(popt)
ARP_map = Am_fit(A0_map, *ARP_fit_coef)
print("ARP_fit_coef:", ARP_fit_coef)

print("min(AG),  max(AG)  =", np.nanmin(AG_map), np.nanmax(AG_map))
print("min(ABP), max(ABP) =", np.nanmin(ABP_map), np.nanmax(ABP_map))
print("min(ARP), max(ARP) =", np.nanmin(ARP_map), np.nanmax(ARP_map))

#%%
#
# Plot sky maps
#

plt.rc('font', size=6.0)
inch = 2.54  # cm
width, height = 8/inch, 17/inch
plt.figure(figsize=(width, height), layout='constrained')

ax = plt.subplot(4, 1, 1)
utils.plot_hpxmap(A0_map, title=r"$A_0$", ax=ax)
plt.gca().set_aspect('equal')

ax = plt.subplot(4, 1, 2)
utils.plot_hpxmap(AG_map, title=r"$A_\mathrm{G}$", ax=ax)
plt.gca().set_aspect('equal')

ax = plt.subplot(4, 1, 3)
utils.plot_hpxmap(ABP_map, title=r"$A_\mathrm{BP}$", ax=ax)
plt.gca().set_aspect('equal')

ax = plt.subplot(4, 1, 4)
utils.plot_hpxmap(ARP_map, title=r"$A_\mathrm{RP}$", ax=ax)
plt.gca().set_aspect('equal')

plt.savefig(config.fig_path+'extinctions.png', dpi=config.fig_dpi)
plt.close()

#%%
#
# Save
#

print("\n* Save extinction maps")

d = {}
d['ag_map']  = AG_map
d['abp_map'] = ABP_map
d['arp_map'] = ARP_map
np.savez_compressed(config.data_path + f"/extinctions.npz", **d)

# %%
