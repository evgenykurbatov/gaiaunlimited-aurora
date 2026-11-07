# %%

import sys

import pandas as pd
import numpy as np

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, ICRS, Galactic, Galactocentric, CartesianDifferential

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

# %%
# Load ARC23 RGB catalog
# Select Aurora RGB stars
#

print("\n* Load ARC23 RGB catalog")

#rgb0 = pd.read_csv('arc23rgb.csv.gz')
rgb0 = pd.read_hdf(config.data_path + '/arc23rgb.hdf', key='rgb')
print(rgb0.columns)
print("Num. of RGB stars:", len(rgb0))

rgb0['hpx'] = rgb0['source_id'] // hpx_base

# %%
# Load GC stars catalogue
#

print("\n* Load GC stars catalogue")

columns = [ 'source_id', 'memberprob' ]
gcstars = pd.read_hdf('GC/vasiliev2021-gcstars.hdf', columns=columns)
print(list(gcstars.columns))

# Leave only high-probability members
mask = gcstars['memberprob'] > 0.9
gcstars = gcstars[mask].copy()
print("records:", len(gcstars))
print("   memory: %.1f MiB" % (gcstars.memory_usage().sum()/(1024**2)))

# %%




# %%
# Make cuts
#

print("\n* Make cuts")

mask = np.full(len(rgb0), True, dtype=bool)
print("  initial records:", mask.sum())

mh_lo_lim = -3.0
mh_hi_lim = 0.5
mask &= (mh_lo_lim <= rgb0['mh_xgboost']) & (rgb0['mh_xgboost'] < mh_hi_lim)
print("  metallicity cut:", mask.sum())

#mask &= rgb0['logg_xgboost'] < config.logg_cut
#print("  log(g) cut:     ", mask.sum())

mask &= ~rgb0['source_id'].isin(gcstars['source_id'])
print("  GC cleaning cut:", mask.sum())

rgb = rgb0[mask].copy()
print("Memory: %.1f MiB" % (rgb.memory_usage().sum()/(1024**2)))
del(rgb0)

# %%




# %%
#
# Transform from Galactic to Galactocentric frame
#

print("\n* Transform from Galactic to Galactocentric frame")

#
# Initialize the Galactocentric frame

gc_frame = Galactocentric(galcen_distance=config.R_sun, z_sun=config.z_sun, galcen_v_sun=config.vel_sun)

x_sun, y_sun, z_sun = config.pos_sun
print("Sun Cartesian pos:", x_sun, y_sun, z_sun)

vx_sun, vy_sun, vz_sun = config.vel_sun
print("Sun Cartesian vel:", vx_sun, vy_sun, vz_sun)

R_sun = np.sqrt(x_sun**2 + y_sun**2)
vR_sun   = (x_sun*vx_sun + y_sun*vy_sun) / config.R_sun
vphi_sun = -(x_sun*vy_sun - y_sun*vx_sun) / config.R_sun
print("Sun Cylindrical vel:", vR_sun, vphi_sun, vz_sun)

#
# Make the transformation

c_icrs = SkyCoord(rgb['ra'].values * u.deg,
                  rgb['dec'].values * u.deg,
                  rgb['dist'].values * u.kpc,
                  pm_ra_cosdec=rgb['pmra'].values * u.mas/u.yr,
                  pm_dec=rgb['pmdec'].values * u.mas/u.yr,
                  radial_velocity=rgb['radial_velocity'].values * u.km/u.s,
                  frame='icrs')

c_gc = c_icrs.transform_to(gc_frame)

x = c_gc.x.to(u.kpc).value
y = c_gc.y.to(u.kpc).value
z = c_gc.z.to(u.kpc).value

vx = c_gc.v_x.to(u.km/u.s).value
vy = c_gc.v_y.to(u.km/u.s).value
vz = c_gc.v_z.to(u.km/u.s).value

rgb['x'], rgb['y'], rgb['z']    =  x,  y,  z
rgb['vx'], rgb['vy'], rgb['vz'] = vx, vy, vz

# %%
#
# Plot
#

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

plt.savefig('fig/arc23aurgb_mh_dist.pdf')
#plt.show()
plt.close()

# %%
#
# Plot
#

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
plt.yscale('log')

plt.subplot(2, 3, 2)
plt.hist(rgb['mg'], bins=100, histtype='step')
plt.xlabel("M_G [mag]")
plt.yscale('log')

plt.subplot(2, 3, 3)
plt.hist(rgb['phot_bp_mean_mag'] - rgb['phot_rp_mean_mag'], bins=100, histtype='step', label="BP - RP")
plt.hist(rgb['phot_bp_mean_mag'] - rgb['phot_rp_mean_mag'] - (rgb['abp'] - rgb['arp']), bins=100, histtype='step', label="BP - RP - (A_BP - A_RP)")
plt.legend(frameon=False)
plt.xlabel("[mag]")
plt.yscale('log')

plt.subplot(2, 3, 4)
plt.hist(rgb['a0'], bins=100, histtype='step', label="A_0")
plt.hist(rgb['ag'], bins=100, histtype='step', label="A_G")
plt.hist(rgb['abp'], bins=100, histtype='step', label="A_BP")
plt.hist(rgb['arp'], bins=100, histtype='step', label="A_RP")
plt.legend(frameon=False)
plt.xlabel("Extinction [mag]")
plt.yscale('log')

plt.subplot(2, 3, 5)
plt.hist2d(rgb['teff_xgboost'], rgb['logg_xgboost'], bins=300, cmap='inferno', norm=LogNorm(), rasterized=True)
plt.xlabel("T_eff [K]")
plt.xlim(3000, 5300)
plt.ylabel("log g")
#plt.ylim(8.5, -8.5)
plt.gca().xaxis.set_inverted(True)
plt.gca().yaxis.set_inverted(True)

plt.subplot(2, 3, 6)
plt.hist2d(rgb['teff_xgboost'], rgb['mg'], bins=300, cmap='inferno', norm=LogNorm(), rasterized=True)
plt.xlabel("T_eff [K]")
plt.xlim(5300, 3000)
plt.gca().xaxis.set_inverted(True)
plt.ylabel("M_G [mag]")
plt.ylim(8.5, -8.5)

plt.savefig("fig/arc23aurgb_magnitudes.pdf")
#plt.show()
plt.close()

# %%
#
# Plot
#

plt.rc('font', size=6.0)
inch = 2.54  # cm
width, height = 8/inch, 6/inch
plt.figure(figsize=(width, height), layout='constrained')

plt.hist2d(rgb['logg_xgboost'], rgb['mg'], bins=100, norm=LogNorm(), cmap='inferno')
plt.plot([3.5], [2.7], marker='*')
plt.xlabel(r"$\log g$")
plt.xlim(xmax=3.75)
plt.ylabel(r"$M_\mathrm{G}$")
plt.ylim(ymax=5.0)

plt.savefig('fig/arc23aurgb_mg_logg.pdf')
#plt.show()
plt.close()

# %%
#
# Save ARC23 Aurora RGB catalog
#

print("\n* Save ARC23 Aurora RGB catalog")

rgb.to_hdf(config.data_path + '/arc23aurgb.hdf', key='au', format='table', data_columns=True, mode='w', index=False)

# %%
