#%%

from tqdm import tqdm
import pickle

import numpy as np
import scipy as sp
from scipy.interpolate import splrep, splev, BSpline
import pandas as pd

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, ICRS, Galactic, Galactocentric

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import global_config as config
import utils


show_figures = False

hpx_order = 7
hpx_base = 2**35 * 4**(12 - hpx_order)
nside = 2**hpx_order
npix = 12 * 4**hpx_order
print(f"hpx_ordfer={hpx_order} -> (nside={nside}, npix={npix})")

#%%
# Load everything

#
# Load ARC23 RGB catalog

print("\n* Load ARC23 RGB catalog")

df = pd.read_hdf(config.data_path + '/arc23rgb.hdf', key='rgb')
N_RGB = len(df)
print("Num. of RGB stars:", N_RGB)

df['hpx'] = df['source_id'] // hpx_base
df['C'] = df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'] - (df['abp'] - df['arp'])

#
# Load model for parallax errors

print("\n* Load model for parallax errors")

with open(config.data_path + '/logplxerr_nvis_g.pkl', 'rb') as f:
    nvis_bins = pickle.load(f)
    plxerr_nvis_g = pickle.load(f)

#
# Load aggregated statistics on parallax errors

print("\n* Load aggregated statistics on parallax errors")

pe0 = pd.read_hdf(config.data_path + '/stats.hdf', key='df')
print("Nans:")
print(pe0.isna().sum())

mask = pe0['g_bin'] >= 0
pe = pe0[mask].copy()
pe.dropna(inplace=True)

del(pe0)

#%%
#
# Choose 'visibility_periods_used' as a relevant feature for hpx-sigma_plx law
#

#pe['nvis_bin'], nvis_bins = pd.qcut(pe['visibility_periods_used'], 10, retbins=True)
nvis_bins = [2, 14, 15, 16, 17, 19, 20, 24, 26, 27.5, 33]
pe['nvis_bin'] = pd.cut(pe['visibility_periods_used'], bins=nvis_bins)
gr = pe.groupby(['nvis_bin', 'g_bin'], observed=True).agg(plxerr=('parallax_error', 'median'))
gr.reset_index(inplace=True)

gr['nvis'] = gr['nvis_bin'].map(lambda x: 0.5*(x.left + x.right))
gr['g'] = 0.5*(config.G_edges[gr['g_bin']] + config.G_edges[gr['g_bin']+1])
print("nvis_bins", nvis_bins)

# %%

"""
plt.figure()
plt.plot(gr['plxerr'], gr['g'], '.')
plt.show()
plt.close()
"""

# %%

gr_ = list(gr.groupby('nvis_bin', observed=True))
lo, hi = gr_[0], gr_[-1]
lo[0], hi[0]

# %%

plxerr_nvis_g

#%%
#
# Plot parallax errors as a function of G and nvis
#

df_ = df

#print("\nPlot parallax errors:")
#print(df_['parallax_error'].describe())

#print("\nBP-RP colors:")
#print(df['C'].describe())

plt.rc('font', size=5.0)
inch = 2.54  ## cm
width, height = 17/inch, 4.15/inch
plt.figure(figsize=(width, height), layout='constrained', dpi=600)
mpl.style.use('tableau-colorblind10')
#style.use('seaborn-v0_8-colorblind')


plt.subplot(1, 3, 1)
plt.hist2d(config.G[pe['g_bin']], pe['parallax_error'], bins=(len(config.G_edges),3000), cmap='Greys', norm=LogNorm(), rasterized=True)
plt.hist2d(df_['phot_g_mean_mag'], df_['parallax_error'], bins=(len(config.G_edges),3000), cmap='inferno', norm=LogNorm(), rasterized=True)
plt.xlabel(r"$\mathrm{G}$ [mag]")
plt.xlim(2.0, 20.0)
plt.xticks(np.arange(2.0, 21.0, 2.0))
plt.ylabel(r"$\sigma_\varpi$ [mas]")
plt.yscale('log')
plt.ylim(6e-3, 4.0)


plt.subplot(1, 3, 2)
#nvis_edges = np.array([2, 3, 5, 8, 13, 21, 33])
nvis_edges = np.array([2, 10, 15, 22, 33])
for i in np.arange(len(nvis_edges)-1)[::-1]:
    mask = (nvis_edges[i] <= pe['visibility_periods_used']) & (pe['visibility_periods_used'] < nvis_edges[i+1])
    pe_ = pe[mask]
    #plt.hist2d(config.G[pe_['g_bin']], pe_['parallax_error'], bins=(len(config.G_edges),3000), cmap='Greys', norm=LogNorm(), rasterized=True)
    plt.plot(config.G[pe_['g_bin']], pe_['parallax_error'], '.', alpha=0.25, label=f"[{nvis_edges[i]}, {nvis_edges[i+1]})")
#plt.hist2d(config.G[pe['g_bin']], pe['parallax_error'], bins=(len(config.G_edges),3000), cmap='Greys', norm=LogNorm(), rasterized=True)
plt.legend(loc='upper left', frameon=True)
plt.xlabel(r"$\mathrm{G}$ [mag]")
plt.xlim(2.0, 20.0)
plt.xticks(np.arange(2.0, 21.0, 2.0))
plt.ylabel(r"$\sigma_\varpi$ [mas]")
plt.yscale('log')
plt.ylim(6e-3, 4.0)


plt.subplot(1, 3, 3)
gr_ = list(gr.groupby('nvis_bin', observed=True))
lo, hi = gr_[0], gr_[-1]
#
plt.plot(config.G[lo[1]['g_bin']], lo[1]['plxerr'], '.', alpha=0.5, label=r"$(2, 14]$")
bin = plxerr_nvis_g[str(lo[0])]
interp = sp.interpolate.interp1d(bin['G'], bin['logsigma_plx'], bounds_error=False, fill_value=(bin['logsigma_plx'][0], bin['logsigma_plx'][-1]))
plt.plot(config.G, 10**interp(config.G), 'k', alpha=0.8)
#
plt.plot(config.G[hi[1]['g_bin']], hi[1]['plxerr'], '.', alpha=0.5, label=r"$(27, 33]$")
bin = plxerr_nvis_g[str(hi[0])]
interp = sp.interpolate.interp1d(bin['G'], bin['logsigma_plx'], bounds_error=False, fill_value=(bin['logsigma_plx'][0], bin['logsigma_plx'][-1]))
plt.plot(config.G, 10**interp(config.G), 'k', alpha=0.8)
#
# Model by Gaia
G_ = config.G.copy()
G_[config.G < 13] = 13
z = 10**(0.4*(G_-15))
plt.plot(config.G, 1e-3*np.sqrt(40 + 800*z + 30*z**2), 'g', alpha=0.5, label="Gaia DR3")
#
plt.legend(loc='center', bbox_to_anchor=(0.48, 0.8), frameon=False)
#plt.axhline(3e-2, ls=':', c='r', alpha=0.5)
#plt.axhline(1e-2, ls=':', c='r', alpha=0.5)
plt.axvline(8.0, ls=':', c='r', alpha=0.5)
plt.axvline(16.0, ls=':', c='r', alpha=0.5)
plt.xlabel(r"$\mathrm{G}$ [mag]")
plt.ylabel(r"$\sigma_\varpi$ [mas]")
plt.yscale('log')
plt.ylim(6e-3, 4e0)


plt.savefig('fig/data-plxerr.png')
if show_figures:  plt.show()
plt.close()

#%%
#
# Make median estimates inside healpixels
#

print("\n* Make median estimates inside healpixels")

plt.rc('font', size=5.0)
inch = 2.54  ## cm
width, height = 8/inch, 7.85/inch
plt.figure(figsize=(width, height), layout='constrained')


peh = pe.groupby('hpx7').agg(nvis=('visibility_periods_used', 'median')).reset_index()
peh['nvis_bin'] = pd.cut(peh['nvis'], bins=nvis_bins).astype(str)

plt.subplot(2, 1, 1)
plt.gca().set_aspect('equal')
utils.plot_hpxmap(peh['nvis'].values, norm=None, xlabel=None, title="visibility_periods_used")


G_ = 10.0
g_bin = np.argmax(np.arange(config.G.size)[config.G <= G_])
sky = np.nan * np.zeros(npix)
for i in tqdm(peh.index):
    r = peh.loc[i,:]
    interp = sp.interpolate.interp1d(plxerr_nvis_g[r['nvis_bin']]['G'], plxerr_nvis_g[r['nvis_bin']]['logsigma_plx'])
    sky[r['hpx7']] = 10**interp(G_)

plt.subplot(2, 1, 2)
plt.gca().set_aspect('equal')
utils.plot_hpxmap(sky, vmax=2e-2, norm=None, title=(r"$\sigma_\varpi$ | G=%g" % G_))


plt.savefig('fig/data-plxerr-sky.pdf')
if show_figures:  plt.show()
plt.close()

# %%
