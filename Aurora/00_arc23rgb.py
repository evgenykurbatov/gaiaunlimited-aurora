#%%

import numpy as np
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

import global_config as config
import utils


#
# Set HEALPix grid parameters
#

hpx_order = 7
hpx_base = 2**35 * 4**(12 - hpx_order)
nside = 2**hpx_order
npix = 12 * 4**hpx_order
print(f"hpx_ordfer={hpx_order} -> (nside={nside}, npix={npix})")

#%%
# Load raw ARC23 RGB catalog
#
# Paper:
#   Andrae R, Rix H-W, Chandra V. Robust Data-driven Metallicities for 175 Million Stars from Gaia XP Spectra (2023)
#   [ADS](https://ui.adsabs.harvard.edu/abs/2023ApJS..267....8A)
#
# Data:
#   [Zenodo](https://zenodo.org/records/7945154)

print("\n* Load raw ARC23 RGB catalog")

# Columns total:
# source_id, l, b, ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error,
# ruwe, radial_velocity, radial_velocity_error,
# phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
# catwise_w1, catwise_w2, mh_xgboost, teff_xgboost, logg_xgboost, in_training_sample

# Load only necessary columns
usecols = ['source_id', 'ra', 'dec', 'l', 'b', 'parallax', 'parallax_error', \
           'pmra', 'pmdec', 'radial_velocity', \
           'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', \
           'mh_xgboost', 'teff_xgboost', 'logg_xgboost']
rgb = pd.read_csv(config.data_path + '/gaia_xp_xgboost_v2.1.data/table_2_catwise.csv.gz', usecols=usecols)
print("Num. of RGB stars:", len(rgb))

#%%
# Distances

# Get the geometric distances
rgb['dist'] = 1.0/rgb['parallax']  # kpc

#%%
# Load dustmap
# Manual: https://dustmaps.readthedocs.io/en/latest/
#
# See also https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_total_galactic_extinction_map.html#total_galactic_extinction_map-a0

print("\n* Load dustmap")

dustmaps.config.config['data_dir'] = config.data_path + '/dustmaps.data'
dustmaps.gaia_tge.fetch()
dm = dustmaps.gaia_tge.GaiaTGEQuery(healpix_level=hpx_order)

#%%
# Get A0 for every star

print("\n* Get A0 for every star")

l = rgb['l'].values
b = rgb['b'].values
co = SkyCoord(l, b, frame='galactic', unit='deg')

rgb['a0'] = dm.query(co)
print("NaNs in A0:", rgb['a0'].isnull().sum())
del(dm)

# Drop stars with no A0
mask = rgb['a0'].isna()
rgb = rgb.loc[~mask,:].copy()
print(f"Stars with A0: {len(rgb)}")

# Plot
plt.title("A0")
l = (rgb['l'] + 180) % 360 - 180
b = rgb['b']
_, _, _, im = plt.hist2d(l, b, weights=rgb['a0'], bins=300, cmap='inferno', norm=LogNorm(), rasterized=True)
plt.colorbar(im)
plt.xlabel("l [deg]")
plt.xlim(180, -180)
plt.ylabel("b [deg]")
plt.savefig("fig/arc23rgb_a0.pdf")
plt.close()

#%%
#
# Estimate AG, ABP, and ARP for every star
#
# Use the Fitzpatrick et al (2019) extinction approximation
#
# Paper:
#   Fitzpatrick E.L., Massa D., Gordon K.D., Bohlin R., Clayton G.C., Dec. 2019, ApJ, 886, 108
#
# Data:
#   https://www.cosmos.esa.int/web/gaia/edr3-extinction-law
#

print("\n* Estimate AG, ABP, and ARP for every star")

# We present another fit adapted for the top of the HR diagram (that is for
# giants and the top of the main sequence, up to MG ∼ 5 mag) where we adapt the
# surface gravity of the Kurucz spectra to the temperature with log g = 4 for
# Teff > 5250 and log g = −8.3 + 0.0023 Teff for cooler stars.

fitz0 = pd.read_csv(config.data_path + '/Fitz19/Fitz19_EDR3_HRDTop.csv')

a0 = rgb['a0']

Teff_0 = 5040.0  # K
teffnorm = rgb['teff_xgboost'] / Teff_0
print(teffnorm.describe())

# See https://www.cosmos.esa.int/web/gaia/edr3-extinction-law
# km = a1 + a2*X + a3*X^2 + a4*X^3 + a5*A0 + a6*A0^2 + a7*A0^3 + a8*A0*X + a9*A0*X^2 + a10*X*A0*2

mask = (fitz0['Xname'] == 'TeffNorm') & (fitz0['Kname'] == 'kG')
fitz = fitz0[mask].to_dict(orient='records')[0]
kG = fitz['Intercept'] + fitz['X']*teffnorm + fitz['X2']*teffnorm**2 + fitz['X3']*teffnorm**3 \
     + fitz['A']*a0 + fitz['A2']*a0**2 + fitz['A3']*a0**3 \
     + fitz['XA']*a0*teffnorm + fitz['AX2']*a0*teffnorm**2 + fitz['XA2']*a0**2*teffnorm
rgb['ag'] = kG * a0

mask = (fitz0['Xname'] == 'TeffNorm') & (fitz0['Kname'] == 'kBP')
fitz = fitz0[mask].to_dict(orient='records')[0]
kBP = fitz['Intercept'] + fitz['X']*teffnorm + fitz['X2']*teffnorm**2 + fitz['X3']*teffnorm**3 \
      + fitz['A']*a0 + fitz['A2']*a0**2 + fitz['A3']*a0**3 \
      + fitz['XA']*a0*teffnorm + fitz['AX2']*a0*teffnorm**2 + fitz['XA2']*a0**2*teffnorm
rgb['abp'] = kBP * a0

mask = (fitz0['Xname'] == 'TeffNorm') & (fitz0['Kname'] == 'kRP')
fitz = fitz0[mask].to_dict(orient='records')[0]
kRP = fitz['Intercept'] + fitz['X']*teffnorm + fitz['X2']*teffnorm**2 + fitz['X3']*teffnorm**3 \
      + fitz['A']*a0 + fitz['A2']*a0**2 + fitz['A3']*a0**3 \
      + fitz['XA']*a0*teffnorm + fitz['AX2']*a0*teffnorm**2 + fitz['XA2']*a0**2*teffnorm
rgb['arp'] = kRP * a0

# Plot
plt.rc('font', size=6.0)
inch = 2.54  # cm
width, height = 24/inch, 6/inch
plt.figure(figsize=(width, height), layout='constrained')

plt.subplot(1, 3, 1)
plt.hist2d(rgb['a0'], rgb['ag'], bins=300, cmap='inferno', norm=LogNorm(), rasterized=True)
#plt.plot(rgb['a0'][::100000], rgb['a0'][::100000], ':k')
plt.xlabel("A_0")
plt.xticks(np.linspace(0.0, 9.0, 10))
plt.ylabel("A_G")
plt.ylim(0.0, 7.0)

plt.subplot(1, 3, 2)
plt.hist2d(rgb['a0'], rgb['abp'], bins=300, cmap='inferno', norm=LogNorm(), rasterized=True)
plt.xlabel("A_0")
plt.xticks(np.linspace(0.0, 9.0, 10))
plt.ylabel("A_BP")
plt.ylim(0.0, 10.0)

plt.subplot(1, 3, 3)
plt.hist2d(rgb['a0'], rgb['arp'], bins=300, cmap='inferno', norm=LogNorm(), rasterized=True)
plt.xlabel("A_0")
plt.xticks(np.linspace(0.0, 9.0, 10))
plt.ylabel("A_RP")
plt.ylim(0.0, 6.0)

plt.savefig("fig/arc23rgb_a0am.pdf")
plt.close()

# %%

# Plot
plt.rc('font', size=6.0)
inch = 2.54  # cm
width, height = 8/inch, 13/inch
plt.figure(figsize=(width, height), layout='constrained')

plt.subplot(2, 1, 1)
plt.hist(rgb['mh_xgboost'], bins=100, histtype='step')
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")

plt.subplot(2, 1, 2)
plt.hist(rgb['dist'], bins=100, histtype='step')
plt.xlabel(r"$D$ [kpc] $= 1/\varpi$ [mas]")

plt.savefig('fig/arc23rgb_mh_dist.pdf')
plt.close()

#%%
#
# Estimate absolute magnitude
#

print("\n* Estimate absolute magnitude")

rgb['mg'] = rgb['phot_g_mean_mag'] - rgb['ag'] - 5*np.log10(rgb['dist']) - 10

# Plot
plt.rc('font', size=6.0)
inch = 2.54  ## cm
width, height = 17/inch, 10/inch
plt.figure(figsize=(width, height), layout='constrained')

plt.subplot(2, 3, 1)
plt.hist(rgb['phot_g_mean_mag'], bins=100, histtype='step', label="G")
plt.hist(rgb['phot_g_mean_mag'] - rgb['ag'], bins=100, histtype='step', label="G - AG")
plt.legend(frameon=False)
plt.xlabel("[mag]")
plt.ylim(ymin=1e2)
plt.yscale('log')

plt.subplot(2, 3, 2)
plt.hist(rgb['mg'], bins=100, histtype='step')
plt.xlabel("M_G [mag]")
plt.ylim(ymin=1e2)
plt.yscale('log')

plt.subplot(2, 3, 3)
plt.hist(rgb['phot_bp_mean_mag'] - rgb['phot_rp_mean_mag'], bins=100, histtype='step', label="BP - RP")
plt.hist(rgb['phot_bp_mean_mag'] - rgb['phot_rp_mean_mag'] - (rgb['abp'] - rgb['arp']), bins=100, histtype='step', label="BP - RP - (A_BP - A_RP)")
plt.legend(frameon=False)
plt.xlabel("[mag]")
plt.ylim(ymin=1e2)
plt.yscale('log')

plt.subplot(2, 3, 4)
plt.hist(rgb['a0'], bins=100, histtype='step', label="A_0")
plt.hist(rgb['ag'], bins=100, histtype='step', label="A_G")
plt.hist(rgb['abp'], bins=100, histtype='step', label="A_BP")
plt.hist(rgb['arp'], bins=100, histtype='step', label="A_RP")
plt.legend(frameon=False)
plt.xlabel("Extinction [mag]")
plt.ylim(ymin=1e2)
plt.yscale('log')

plt.subplot(2, 3, 5)
plt.hist2d(rgb['teff_xgboost'], rgb['logg_xgboost'], bins=300, cmap='inferno', norm=LogNorm(), rasterized=True)
plt.xlabel("T_eff [K]")
plt.xlim(3000, 5200)
plt.ylabel("log g")
#plt.ylim(8.5, -8.5)

plt.subplot(2, 3, 6)
plt.hist2d(rgb['teff_xgboost'], rgb['mg'], bins=300, cmap='inferno', norm=LogNorm(), rasterized=True)
plt.xlabel("T_eff [K]")
plt.xlim(5200, 3000)
plt.ylabel("M_G [mag]")
plt.ylim(8.5, -8.5)

plt.savefig("fig/arc23rgb_magnitudes.pdf")
plt.close()

#%%
#
# Save ARC23 Aurora RGB catalog
#

print("\n* Save ARC23 RGB catalog")

print("Check for NaNs:")
print(rgb.isna().sum())

rgb.to_hdf('arc23rgb.hdf5', key='rgb', format='table', data_columns=True, mode='w', index=False)
