# %%

from tqdm import tqdm
import pickle

import numpy as np
import scipy as sp
from scipy.interpolate import splrep, splev, BSpline
import pandas as pd

#import astropy.units as u
#from astropy_healpix import HEALPix
#from astropy.coordinates import SkyCoord, ICRS, Galactic, Galactocentric

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import config
import utils


G_edges = config.G_edges

# %%

hpx_order = 7
hpx_base = 2**35 * 4**(12 - hpx_order)
nside = 2**hpx_order
npix = 12 * 4**hpx_order
print(f"hpx_ordfer={hpx_order} -> (nside={nside}, npix={npix})")

# %%
# Load ARC23 RGB catalog

print("\n* Load ARC23 RGB catalog")

df = pd.read_hdf(config.data_path + '/arc23rgb.hdf', key='rgb')
print(df.columns)
print("Num. of RGB stars:", len(df))

# %%
# Load aggregated statistics on parallax errors

print("\n* Load aggregated statistics on parallax errors")

stats0 = pd.read_hdf(config.data_path + '/stats.hdf', key='df')
print("Nans:")
print(stats0.isna().sum())

mask = stats0['g_bin'] >= 0
stats = stats0[mask].copy()
stats.dropna(inplace=True)

del(stats0)

# %%



# %%
# Plot parallax errors as a function of G and MG

df_ = df

print("\nPlot parallax errors:")
print(df_['parallax_error'].describe())

clr = df_['phot_bp_mean_mag'] - df_['phot_rp_mean_mag']
print("\nBP-RP colors:")
print(clr.describe())

plxerr_edges = np.logspace(-3.0, 1.0, 101)


plt.rc('font', size=6.0)
inch = 2.54  ## cm
width, height = 17/inch, 13/inch
plt.figure(figsize=(width, height), layout='constrained')
cmap = 'inferno'
norm = 'log'

plt.subplot(2, 2, 1)
plt.title("All Gaia stars")
plt.hist2d(G_edges[stats['g_bin']], stats['parallax_error'], bins=(G_edges, plxerr_edges),
           cmap=cmap, norm=norm, rasterized=True)
plt.xlabel(r"$G$ [mag]")
plt.xlim(G_edges[0], G_edges[-1])
plt.ylabel(r"$\sigma_\varpi$ [mas]")
plt.yscale('log')
plt.ylim(3.33e-3, 4.0)

plt.subplot(2, 2, 2)
plt.title("ARC23 RGB stars")
plt.hist2d(df_['phot_g_mean_mag'], df_['parallax_error'], bins=(G_edges, plxerr_edges),
           cmap=cmap, norm=norm, rasterized=True)
plt.xlabel(r"$G$ [mag]")
plt.xlim(G_edges[0], G_edges[-1])
plt.ylabel(r"$\sigma_\varpi$ [mas]")
plt.yscale('log')
plt.ylim(3.33e-3, 4.0)

plt.subplot(2, 2, 3)
C_bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
for i in range(len(C_bins)-1):
    clr_mask = (C_bins[i] < clr) & (clr <= C_bins[i+1])
    plt.plot(df_[clr_mask]['phot_g_mean_mag'], df_[clr_mask]['parallax_error'],
             '.', alpha=0.25, label=f"BP-RP in ({C_bins[i]}, {C_bins[i+1]}]")
plt.legend()
plt.xlabel(r"$G$ [mag]")
plt.ylabel(r"$\sigma_\varpi$ [mas]")
plt.yscale('log')

plt.subplot(2, 2, 4)
plt.title("G in (14.0, 14.2]")
g_mask = (14.0 < df_['phot_g_mean_mag']) & (df_['phot_g_mean_mag'] <= 14.2)
clr = df_['phot_bp_mean_mag'] - df_['phot_rp_mean_mag']
#C_bins = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
C_bins = np.array([1.0, 2.0, 3.0, 4.0])
for i in range(len(C_bins)-1):
    clr_mask = (C_bins[i] < clr) & (clr <= C_bins[i+1])
    mask = g_mask & clr_mask
    plt.hist(df_[mask]['parallax_error'], bins=100, range=(0.0, 0.1),
             histtype='step', label=f"BP-RP in ({C_bins[i]}, {C_bins[i+1]}]")
plt.legend()
plt.xlabel(r"$\sigma_\varpi$ [mas]")
plt.xlim(xmax=0.04)
#plt.yscale('log')

plt.savefig(config.fig_path + 'data_plxerr_stats.png', dpi=config.fig_dpi)

# %%
# Delete what's not needed

del(df)

# %%



# %%
# Choose 'visibility_periods_used' as a relevant feature for hpx-sigma_plx law

#pe['nvis_bin'], nvis_bins = pd.qcut(pe['visibility_periods_used'], 10, retbins=True)
nvis_bins = [2, 14, 15, 16, 17, 19, 20, 24, 26, 27.5, 33]
stats['nvis_bin'] = pd.cut(stats['visibility_periods_used'], bins=nvis_bins)
gr = stats.groupby(['nvis_bin', 'g_bin'], observed=True).agg(plxerr=('parallax_error', 'median'))
gr.reset_index(inplace=True)

gr['nvis'] = gr['nvis_bin'].map(lambda x: 0.5*(x.left + x.right))
gr['g'] = 0.5*(G_edges[gr['g_bin']] + G_edges[gr['g_bin']+1])
print("nvis_bins", nvis_bins)

# %%
# Fit parallax errors by G given nvis

def fit(G, plxerr):
    x = G
    y = np.log10(plxerr)
    s = len(x)
    tck = splrep(x, y, s=s)
    return tck

plxerr_nvis_g = {}

G_min = 8.0
G_max = G_edges.max()
G_edges_ = np.linspace(G_min, G_max, 17)

for bin, gr_ in gr.groupby('nvis_bin', observed=True):
    tck = fit(gr_['g'], gr_['plxerr'])
    plxerr_nvis_g[str(bin)] = {}
    plxerr_nvis_g[str(bin)]['G'] = G_edges_
    plxerr_nvis_g[str(bin)]['logsigma_plx'] = BSpline(*tck)(G_edges_)

# %%
# Plot

plt.rc('font', size=6.0)
inch = 2.54  ## cm
width, height = 30/inch, 11/inch
plt.figure(figsize=(width, height), layout='constrained')
plt.suptitle(r"$G$ [mag] vs $\sigma_\varpi$ [mas]")

i = 0
for bin, gr_ in gr.groupby('nvis_bin', observed=True):
    plt.subplot(2, 5, i+1)

    plt.title(f"nvis: {bin}")

    G = G_edges[gr_['g_bin']]
    plt.plot(G, gr_['plxerr'], ls='', marker='.', alpha=0.5)

    #plt.plot(plxerr_nvis_g[str(bin)]['G'], 10**plxerr_nvis_g[str(bin)]['logsigma_plx'], 'k')
    plt.plot(G, 10**np.interp(G, plxerr_nvis_g[str(bin)]['G'], plxerr_nvis_g[str(bin)]['logsigma_plx']), 'k')
    plt.axhline(3e-2, ls=':', c='r', alpha=0.5)
    plt.axhline(1e-2, ls=':', c='r', alpha=0.5)
    plt.axvline(8.0, ls=':', c='k', alpha=0.5)
    plt.axvline(16.0, ls=':', c='k', alpha=0.5)
    plt.xlabel(r"$G$ [mag]")
    plt.yscale('log')
    plt.ylim(3.33e-3, 4.0)

    i += 1

plt.savefig('fig/plxerr_nvis_g.pdf')
plt.close()

# %%
#
# Make median estimates inside healpixels

print("\n* Make median estimates inside healpixels")

peh = stats.groupby('hpx7').agg(nvis=('visibility_periods_used', 'median')).reset_index()
peh['nvis_bin'] = pd.cut(peh['nvis'], bins=nvis_bins).astype(str)

plt.figure()
utils.plot_hpxmap(peh['nvis'].values, norm=None, title='visibility_periods_used')
plt.savefig('fig/nvis_sky.pdf')

# Plot
G_ = 10.0
g_bin = np.argmax(np.arange(G_edges.size)[G_edges <= G_])
sky = np.nan * np.zeros(npix)
for i in tqdm(peh.index):
    r = peh.loc[i,:]
    interp = sp.interpolate.interp1d(plxerr_nvis_g[r['nvis_bin']]['G'], plxerr_nvis_g[r['nvis_bin']]['logsigma_plx'])
    sky[r['hpx7']] = 10**interp(G_)

plt.figure()
utils.plot_hpxmap(sky, norm=None, title=f"sigma_varpi for G={G_}")
plt.savefig('fig/plxerr_g=10_sky.pdf')

# %%
# Save

print("\n* Save model for parallax errors")

with open(config.data_path + '/logplxerr_nvis_g.pkl', 'wb') as f:
    pickle.dump(nvis_bins, f)
    pickle.dump(plxerr_nvis_g, f)

print("\n* Done")

# %%
