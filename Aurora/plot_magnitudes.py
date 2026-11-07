# %%

import sys

import pandas as pd
import numpy as np
import scipy as sp

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, ICRS, Galactic, Galactocentric

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

import global_config as config
import utils

#
# Set HEALPix grid parameters
#

hpxorder = 7
hpxbase = 2**35 * 4**(12 - hpxorder)
nside = 2**hpxorder
npix = 12 * 4**hpxorder
print(f"hpxorder={hpxorder} -> (nside={nside}, npix={npix})")

# %%
# Load everything
#

#
# Load ARC23 RGB catalog

print("\n* Load ARC23 RGB catalog")

columns = ['source_id', 'a0', 'ag', 'abp', 'arp', 'teff_xgboost', 'mh_xgboost', 'logg_xgboost']
xp = pd.read_hdf(config.data_path + '/arc23rgb.hdf', key='rgb', columns=columns)
print(xp.columns)
print("Num. of RGB stars:", len(xp))

xp['hpx'] = xp['source_id'] // hpxbase

#
# Load ARC23 Aurora RGB catalog

print("\n* Load ARC23 Aurora RGB catalog")

columns = ['hpx', 'x', 'y', 'z', 'dist', 'phot_g_mean_mag', 'mg', 'ag', 'teff_xgboost', 'mh_xgboost', 'logg_xgboost']
au = pd.read_hdf(config.data_path + '/arc23aurgb.hdf', key='au', columns=columns)
print(au.columns)
print("Num. of Aurora RGB stars:", len(au))

# Downscale healpixel indices if necessary
if hpxorder < 7:
    au['hpx'] = au['hpx'] // 4**(7-hpxorder)

print(au['mh_xgboost'].describe())

# %%
# Select stars by the distance
#

#mask = (0.01 < au['dist']) & (au['dist'] < 18.0)
mask = 0.0 < au['dist']
au = au[mask].copy()
N_AuRGB = len(au)
print("Num. of Aurora RGB stars after selection by distance:", N_AuRGB)

# Calculate the Galactocentric distance
au['r'] = np.sqrt(au['x']**2 + au['y']**2 + au['z']**2)

# %%
# Select stars by metallicity and log(g)
#

print("\n* Select stars by metallicity and log(g)")

mh_lo_lim = -3.0
mh_hi_lim = -1.0
mask = (mh_lo_lim <= au['mh_xgboost']) & (au['mh_xgboost'] < mh_hi_lim)
print("  metallicity cut:", mask.sum())
au = au[mask].copy()

logg_mask = au['logg_xgboost'] < config.logg_cut
print("  logg_cut =", config.logg_cut)
print("  log(g) cut:     ", logg_mask.sum())

#%%
# Fit extinctions
#

print("\n* Fit extinctions")

def Am_fit(A0, c1, c2, c3, c4):
    return c1*A0 + c2*A0**2 + c3*A0**3 + c4*A0**4

# G
popt, _ = sp.optimize.curve_fit(Am_fit, xp['a0'], xp['ag'])
AG_fit_coef = list(popt)
print("AG_fit_coef: ", AG_fit_coef)

# BP
popt, _ = sp.optimize.curve_fit(Am_fit, xp['a0'], xp['abp'])
ABP_fit_coef = list(popt)
print("ABP_fit_coef:", ABP_fit_coef)

# RP
popt, _ = sp.optimize.curve_fit(Am_fit, xp['a0'], xp['arp'])
ARP_fit_coef = list(popt)
print("ARP_fit_coef:", ARP_fit_coef)

# %%
# Plot
#

# Plot
plt.rc('font', size=5.0)
inch = 2.54  ## cm
width, height = 17/inch, 9/inch
plt.figure(figsize=(width, height), layout='constrained')

#mpl.style.use('tableau-colorblind10')
#mpl.style.use('seaborn-v0_8-colorblind')
sns.color_palette('colorblind')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

#plt.suptitle(r"Andrae et al. (2023) RGB stars | $-3 < \mathrm{[Fe/H]} < -1.3$")

alpha = 1.0

N_au = len(au)
N_au_mask = len(au[logg_mask])

plt.subplot(2, 3, 1)
#
hist, edges = np.histogram(au['phot_g_mean_mag'], bins=30, density=True, range=(6.0, 16.0))
plt.stairs(N_au*hist, edges, ls='--', color=colors[0], alpha=alpha)
hist, _ = np.histogram(au[logg_mask]['phot_g_mean_mag'], bins=edges, density=True, range=(6.0, 16.0))
plt.stairs(N_au_mask*hist, edges, ls='-', color=colors[0], alpha=alpha, label=r"$\mathrm{G}$")
#
hist, edges = np.histogram(au['phot_g_mean_mag'] - au['ag'], bins=30, density=True, range=(6.0, 16.0))
plt.stairs(N_au*hist, edges, ls='--', color=colors[3], alpha=alpha)
hist, _ = np.histogram(au[logg_mask]['phot_g_mean_mag'] - au[logg_mask]['ag'], bins=edges, density=True, range=(6.0, 16.0))
plt.stairs(N_au_mask*hist, edges, ls='-', color=colors[3], alpha=alpha, label=r"$\mathrm{G} - A_\mathrm{G}$")
#
plt.legend(frameon=False)
plt.xlabel(r"$\mathrm{G}$, $\mathrm{G} - A_\mathrm{G}$ [mag]")
plt.xlim(7.0, 17.0)
plt.yscale('log')
plt.ylim(ymin=1e2)


plt.subplot(2, 3, 2)
#
hist, edges = np.histogram(au['mg'], bins=30, density=True, range=(-6.0, 4.0))
plt.stairs(N_au*hist, edges, ls='--', color=colors[0], alpha=alpha)
hist, _ = np.histogram(au[logg_mask]['mg'], bins=edges, density=True, range=(-6.0, 4.0))
plt.stairs(N_au_mask*hist, edges, color=colors[0], alpha=alpha)
#
#plt.legend(frameon=False)
plt.xlabel(r"$\mathrm{M}_\mathrm{G}$ [mag]")
plt.xlim(-6.0, 4.0)
plt.yscale('log')
plt.ylim(ymin=1e2)


plt.subplot(2, 3, 3)
a0 = np.linspace(0.0, 10.0, 201)
plt.hist2d(xp['a0'], xp['ag'], bins=300, cmap='Greens', norm=LogNorm(), rasterized=True)
plt.plot(a0, Am_fit(a0, *AG_fit_coef), 'g', lw=1.2, alpha=0.7)
plt.xlabel(r"$A_0$ [mag]")
plt.xticks(np.arange(0.0, 11.0, 2.0))
plt.ylabel(r"$A_\mathrm{G}$ [mag]")
plt.ylim(0.0, 7.0)


plt.subplot(2, 3, 4)
plt.hist2d(xp['teff_xgboost'], xp['logg_xgboost'], bins=300, cmap='Greys', norm='log', rasterized=True)
plt.hist2d(au['teff_xgboost'], au['logg_xgboost'], bins=300, cmap='inferno', norm='log', rasterized=True)
plt.xlabel(r"$T_\mathrm{eff}$ [K]")
plt.xlim(5200, 3000)
plt.ylabel(r"$\log(g)$")
plt.gca().invert_yaxis()
#plt.ylim(config.logg_cut, -0.5)


#plt.subplot(2, 3, 5)
#plt.hist2d(xp['teff_xgboost'], xp['mg'], bins=300, cmap='Greys', norm=LogNorm(), rasterized=True)
#plt.hist2d(au['teff_xgboost'], au['mg'], bins=300, cmap='inferno', norm=LogNorm(), rasterized=True)
#plt.xlabel(r"$T_\mathrm{eff}$ [K]")
#plt.xlim(5200, 3000)
#plt.ylabel(r"$M_\mathrm{G}$ [mag]")
#plt.ylim(5.0, -8.5)

plt.subplot(2, 3, 5)
#
hist, edges = np.histogram(au['dist'], bins=30, density=True, range=(0.0, 20.0))
plt.stairs(N_au*hist, edges, ls='--', color=colors[0], alpha=alpha)
hist, _ = np.histogram(au[logg_mask]['dist'], bins=edges, density=True)
plt.stairs(N_au_mask*hist, edges, ls='-', color=colors[0], alpha=alpha)
#
plt.xlabel(r"$D$ [kpc]")
plt.yscale('log')
plt.ylim(ymin=1e2)


plt.subplot(2, 3, 6)
hist, edges = np.histogram(xp['mh_xgboost'], bins=75)
#plt.stairs(len(xp)*np.cumsum(hist), edges, alpha=alpha)
plt.plot(edges[:-1], np.cumsum(hist) / np.sum(hist), 'k', alpha=alpha)
plt.axvline(-3.0, ls=':', c='gray')
plt.axvline(-2.0, ls=':', c='gray')
plt.axvline(-1.5, ls=':', c='gray')
plt.axvline(-1.3, ls=':', c='gray')
plt.axvline(-1.0, ls=':', c='gray')
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
plt.xlim(xmin=edges[0])
plt.ylabel(r"Fraction of stars $< [\mathrm{M}/\mathrm{H}]$")
plt.yscale('log')
#plt.ylim(ymin=1.0)


plt.savefig("fig/data-magnitudes.pdf")
#plt.show()
plt.close()
