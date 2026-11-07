# %%

import pickle
from tqdm import tqdm
import pandas as pd

import numpy as np
import scipy as sp
from scipy.special import expit, logit

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord

import gaiaunlimited

import matplotlib.pyplot as plt

import global_config as config
from domain import Domain
import utils

# %%
# Load domain
#

print("\n* Load domain")

dom = Domain().load(config.data_path + f"/domain_hpx{config.hpxorder}.npz")

# %%
# Load GS statistics
#

print("\n* Load GS statistics")

gs = pd.read_hdf(config.data_path + '/stats.hdf')
print(gs.columns.tolist())

gs = gs[['hpx7', 'g_bin', 'n', 'k']]
print("Num. of GS stars:", gs['n'].sum())
# Clean
mask = gs['g_bin'] == -9999
gs = gs[~mask]
print("stars after cleaning:  ", gs['n'].sum())
print("records after cleaning:", len(gs))
print("   memory: %.1f MiB" % (gs.memory_usage().sum()/(1024**2)))

# %%



# %%
# Apply the Hierarchical MLE for SSF

hplevel_and_binning = { 'healpix': 7,
                        'phot_g_mean_mag': [dom.G_edges[0], dom.G_edges[-1], dom.dG[0]] }

gs.rename(columns={'hpx7': 'ipix', 'g_bin': 'phot_g_mean_mag_'}, inplace=True)

ssf_stats = gaiaunlimited.subsample.SubsampleSelectionFunctionHMLE().use(gs, hplevel_and_binning=hplevel_and_binning)

# %%
# Plot sky map

# Collect the total number of sources inside every healpix, _regardless their magnitudes_ (the summation over the last axis is for that)
ds = ssf_stats.hds[dom.hpxorder]
n = ds['n'].sum(axis=-1).values
k = ds['k'].sum(axis=-1).values
prb = k/n
print("shapes: ", n.shape, k.shape, prb.shape)

plt.rc('font', size=5.0)
inch = 2.54  ## cm
width, height = 8/inch, 11.8/inch
plt.figure(figsize=(width, height), layout='constrained')

plt.subplot(3, 1, 1)
plt.gca().set_aspect('equal')
utils.plot_hpxmap(n, interp=False, xlabel=None, title=f"No. GS sources per HEALPix {dom.hpxorder}")

plt.subplot(3, 1, 2)
plt.gca().set_aspect('equal')
utils.plot_hpxmap(k, interp=False, xlabel=None, title=f"No. XP sources per HEALPix {dom.hpxorder}")

plt.subplot(3, 1, 3)
plt.gca().set_aspect('equal')
utils.plot_hpxmap(prb, interp=False, norm=None, title=f"Subsample SF of XP/GS")

plt.savefig(f"fig/data-sf-xp-sky-hpx{dom.hpxorder}.pdf")
#plt.show()
plt.close()

# %%
# Plot piercings

ds = ssf_stats.hds[dom.hpxorder]
n = ds['n']
k = ds['k']
prb = expit(ds['logitp'])

# Smooth probability with Gaussian
width = 2.0
window = sp.signal.windows.gaussian(len(dom.G), std=width) / (np.sqrt(2*np.pi)*width)
prb_smo = np.array([ sp.signal.convolve(prb_, window, mode='same')  for prb_ in prb ])

# Sky-global probability
prb_glob = k.sum(axis=0) / n.sum(axis=0)

print("shapes: ", n.shape, k.shape, prb.shape, prb_smo.shape, prb_glob.shape)


plt.rc('font', size=5.0)
inch = 2.54  ## cm
width, height = 8/inch, 7.4/inch
plt.figure(figsize=(width, height), layout='constrained')
#plt.suptitle("Subsample SF of XP/GS")

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
alpha = 0.8


# Select a target direction
#l, b = 75.0, -45.0
l, b = 75.0, -40.0
#l, b = -60.0, -30.0
#l, b = -30.0, -60.0
#l, b = -10.0, -60.0
co = SkyCoord(l, b, frame='galactic', unit='deg')
hp = HEALPix(nside=dom.nside, order='nested', frame='icrs')
hpx = hp.skycoord_to_healpix(co)
print("l, b, hpx =", (l, b, hpx))

plt.subplot(2, 2, 1)
plt.title((r"$(l, b) = (%g^\circ, %g^\circ)$ [deg]" % (l, b)))
plt.plot(dom.G, n[hpx], c='k', alpha=alpha, label="n")
plt.plot(dom.G, k[hpx], c='grey', alpha=alpha, label="k")
plt.legend(frameon=False)
plt.xlabel("G [mag]")
plt.xlim(xmax=20)
plt.ylim(ymin=1.0)
plt.yscale('log')

plt.subplot(2, 2, 3)
plt.plot(dom.G, prb_glob, 'k', alpha=0.5*alpha)
plt.plot(dom.G, prb[hpx], c=colors[1], alpha=alpha)
plt.plot(dom.G, prb_smo[hpx], c=colors[0], alpha=alpha)
plt.xlabel("G [mag]")
plt.xlim(xmax=20)
#plt.xlim(xmin=5, xmax=20)


# Select a target direction
l, b = 135.0, 41.0
co = SkyCoord(l, b, frame='galactic', unit='deg')
hp = HEALPix(nside=dom.nside, order='nested', frame='icrs')
hpx = hp.skycoord_to_healpix(co)

plt.subplot(2, 2, 2)
plt.title((r"$(l, b) = (%g^\circ, %g^\circ)$ [deg]" % (l, b)))
plt.plot(dom.G, n[hpx], c='k', alpha=alpha, label="n")
plt.plot(dom.G, k[hpx], c='grey', alpha=alpha, label="k")
plt.legend(frameon=False)
plt.xlabel("G [mag]")
plt.xlim(xmax=20)
plt.ylim(ymin=1.0)
plt.yscale('log')

plt.subplot(2, 2, 4)
plt.plot(dom.G, prb_glob, 'k', alpha=0.5*alpha)
plt.plot(dom.G, prb[hpx], c=colors[1], alpha=alpha)
plt.plot(dom.G, prb_smo[hpx], c=colors[0], alpha=alpha)
plt.xlabel("G [mag]")
plt.xlim(xmax=20)
#plt.xlim(xmin=5, xmax=20)


plt.savefig(f"fig/data-sf-xp-hpx{dom.hpxorder}.pdf")
#plt.show()
plt.close()

# %%




# %%
# Save
#

print("\n* Save")

ds = ssf_stats.hds[dom.hpxorder]
prb = expit(ds['logitp'])
# Smooth probability with Gaussian
width = 2.0
window = sp.signal.windows.gaussian(len(dom.G), std=width) / (np.sqrt(2*np.pi)*width)
prb_smo = np.array([ sp.signal.convolve(prb_, window, mode='same')  for prb_ in prb ])

d = {}
d['prb_xp'] = prb
d['prb_xp_smo'] = prb_smo
fpath = config.data_path + f"/selfun_xp_hpx{dom.hpxorder}.npz"
np.savez_compressed(fpath, **d)

# %%
