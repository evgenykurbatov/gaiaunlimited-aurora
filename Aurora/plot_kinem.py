# %%

import os
import sys
import pathlib
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, ICRS, Galactic, Galactocentric, CartesianDifferential

import agama

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn as sns

import global_config as config
#import utils

show_figures = False

# %%
# Load everything

#
# Load ARC23 Aurora RGB catalog

print("\n* Load ARC23 Aurora RGB catalogue")

hdf = pd.HDFStore(config.data_path + '/arc23aurgb.hdf')
nrows = hdf.get_storer('au').shape
print("Num. of RGB stars:", nrows)

# Metallicity range to select
mh_lo_lim, mh_hi_lim = -3.0, 0.5

df = pd.DataFrame()
chunksize = 10**6
for i in tqdm(range(nrows//chunksize + 1)):
    usecols = ['source_id', 'l', 'b', 'parallax', 'parallax_error', 'dist', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'mh_xgboost']
    #usecols = ['dist', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'mh_xgboost']
    tmp = hdf.select('au', start=i*chunksize, stop=(i+1)*chunksize)[usecols]

    # Select stars by metallicity
    mask = (mh_lo_lim <= tmp['mh_xgboost']) & (tmp['mh_xgboost'] < mh_hi_lim)
    tmp = tmp[mask].copy()

    # For finer slection
    tmp['parallax_over_error'] = tmp['parallax'] / tmp['parallax_error']
    tmp.drop(columns=['parallax', 'parallax_error'], inplace=True)

    df = pd.concat([df, tmp], ignore_index=True)

del(tmp)
print("Num. of RGB stars after metallicity cut:", len(df))

#
# Load the catalogue of Vasiliev&Baumgardt (2021, MNRAS 505, 5978)
# Data: https://doi.org/10.5281/zenodo.4891252

print("\n* Load the GC stars catalogue")

gc_data_path = config.data_root_path + '/Vasiliev21/'
fnames = [ f.name  for f in pathlib.Path(gc_data_path + '/catalogues').glob(f"*.txt") ]
print("len(fnames) =", len(fnames))

gc = pd.DataFrame()
for fname in fnames:
    gc_ = pd.read_csv(gc_data_path + 'catalogues/' + fname, delimiter='\s+', comment='#',
                      names=['source_id', 'ra', 'dec', 'x', 'y', 'plx', 'pmra', 'pmdec', 'plxe', 'pmrae', 'pmdece', 'pmcorr',
                             'g_mag', 'bp_rp', 'Sigma', 'qflag', 'memberprob'],
                      dtype={'source_id': int, 'qflag': int})
    gc = pd.concat([gc, gc_[['source_id', 'memberprob']]], ignore_index=True)
print("Num. of GC stars:", len(gc))

# %%
# Clean

print("\n* Clean")

#
# Select stars by the distance
#mask = (0.01 < df['dist']) & (df['dist'] < 18.0)
#df = df[mask].copy()
#print("Num. of Aurora RGB stars after selection by distance:", len(au))

# Exclude NaNs
mask = df['vx'].isna() | df['vy'].isna() | df['vz'].isna()
df = df[~mask].copy()

print("Num. of stars after cleaning:", len(df))

# %%
# More cleaning

"""
# Get more strict parallaxes
mask = df['parallax'] > 10*df['parallax_error']
df = df[mask].copy()

# Remove the very Galactic plane
mask = np.abs(df['b']) < 10.0
df = df[~mask].copy()

mask = np.fabs(df['z']) < 2.5
df = df[mask].copy()

print("Num. of stars after more cleaning:", len(df))
"""

# %%
# Transform from Cartesian to Cylindrical frame

# Galactocentric distance in the Spherical coordinates
#df['r'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
# Galactocentric distance in the Cylindrical coordinates
df['R'] = np.sqrt(df['x']**2 + df['y']**2)

# Galactocentric velocities in Cylindrical coordinates
df['vR']   = (df['x']*df['vx'] + df['y']*df['vy']) / df['R']
df['vphi'] = -(df['x']*df['vy'] - df['y']*df['vx']) / df['R']

# Angular momentum
df['Lz'] = df['R'] * df['vphi']

# %%
# Estimate the model-dependent kinematics
#
# Galaxy model
# References: Sormani et al. 2022 (MNRAS Letters/514/L1); Hunter et al. 2024 (submitted); Vasiliev 2018 (1802.08255)

print("\n* Estimate the model-dependent kinematics")

kinem_file = config.data_path + '/arc23aurgb_kinem.hdf'
if os.path.exists(kinem_file):
    print("Load kinematics")
    df = pd.read_hdf(kinem_file, key='df')
else:
    print("Calculate kinematics")

    #pot = agama.Potential('agama_MWPotentialHunter24_full.ini')
    pot = agama.Potential('agama_MWPotentialHunter24_rotating.ini')

    with agama.setNumThreads(12):
        print("E ...")
        df['E'] = 0.5*(df['vx']**2 + df['vy']**2 + df['vz']**2) + pot.potential(df[['x', 'y', 'z']])
        print("periapo ...")
        periapo = pot.Rperiapo(df[['E', 'Lz']])
        df['peri'] = periapo[:,0]
        df['apo']  = periapo[:,1]
        print("rcirc ...")
        df['rcirc'] = pot.Rcirc(E=df['E'])
        print("tcirc ...")
        df['tcirc'] = pot.Tcirc(df['E'])

    # Angular momentum on circular orbits
    df['Lc'] = 2*np.pi*df['rcirc']**2 / df['tcirc']
    df['eta'] = df['Lz']/df['Lc']

    # Eccentricity
    x = df['apo']/df['peri']
    df['ecc'] = (x - 1) / (x + 1)

    # Save kinematics
    df.to_hdf(kinem_file, key='df', mode='w', index=False)

# %%
# Cut SMC and LMC

print("\n* Cut SMC and LMC")

# SMC
co = SkyCoord(16.26*u.deg, -72.42*u.deg, frame='icrs')
l0, b0 = co.galactic.l.value, co.galactic.b.value
mask = np.sqrt((df['l']-l0)**2 + (df['b']-b0)**2) < 8.0
df = df[~mask].copy()

# LMC
co = SkyCoord(78.77*u.deg, -69.01*u.deg, frame='icrs')
l0, b0 = co.galactic.l.value, co.galactic.b.value
mask = np.sqrt((df['l']-l0)**2 + (df['b']-b0)**2) < 12.0
df = df[~mask].copy()

print("Num. of stars after SMC+LMC cut:", len(df))

# %%
# Cut GC stars

print("\n* Cut GC stars")

mask = ~df['source_id'].isin(gc['source_id'])
df = df[mask].copy()

print("Num. of stars after GC cut:", len(df))

# %%
# Drop what's no longer needed

#df.drop(columns=['x', 'y', 'z', 'vx', 'vy', 'vz'], inplace=True)

# %%
# Bin

print("\n* Bin")

mask = df['ecc'].isna()
df = df[~mask].copy()

mh_nbins = 91
other_nbins = 61

df['mh_'], mh_edges   = pd.cut(df['mh_xgboost'], bins=mh_nbins, retbins=True)
df['Lz_'], Lz_edges   = pd.cut(df['Lz']/1e3, bins=np.linspace(-2.0, 4.0, other_nbins), retbins=True)
df['E_'], E_edges   = pd.cut(df['E']/1e5, bins=np.linspace(-2.4, -0.8, other_nbins), retbins=True)
df['eta_'], eta_edges = pd.cut(df['eta'],    bins=np.linspace(-1.0, 1.0, other_nbins), retbins=True)
#df['apo_'], apo_edges = pd.cut(df['apo'],    bins=np.linspace(0.0, 10**1.5, other_nbins), retbins=True)
#df['apo_'], apo_edges = pd.cut(df['apo'],    bins=np.logspace(np.log10(0.5), 1.5, other_nbins), retbins=True)
df['apo_'], apo_edges = pd.cut(df['apo'],    bins=np.logspace(np.log10(2.33), 1.5, other_nbins), retbins=True)
df['ecc_'], ecc_edges = pd.cut(df['ecc'],    bins=np.linspace(0.0, 1.0, other_nbins), retbins=True)

def plot_hist2d(df, x, y, map_, x_edges, y_edges, ax=None, colnorm=True, maskzeros=True, cb=False, **kwargs):
    if ax is None:
        ax = plt.gca()
    tmp = df.groupby([x, y], observed=False).agg(val=map_)
    val = pd.DataFrame(tmp).reset_index().pivot(index=y, columns=x, values='val').astype(float).values
    print("shape(val) =", val.shape)
    if colnorm:
        ####val_min = val.min(axis=0)[np.newaxis,:]
        ####val_max = val.max(axis=0)[np.newaxis,:]
        ####val = (val - val_min) / (val_max - val_min)
        val /= val.sum(axis=0)
    if maskzeros:
        val[np.where(val == 0)] = np.nan
    im = ax.pcolormesh(x_edges, y_edges, val, rasterized=True, **kwargs)
    if cb:
        cbar = plt.colorbar(im, ax=ax)
        return cbar

# %%



# %%
# 2-D histograms

df_ = df
#mask = df['parallax_over_error'] > 10
#df_ = df[mask]


plt.rc('font', size=5.0)
inch = 2.54  ## cm
#width, height = 17/inch, 5.69/inch
width, height = 17/inch, 3.5/inch
plt.figure(figsize=(width, height), layout='constrained')

plt.subplot(1, 4, 1)
cmap = 'gist_heat_r'
plot_hist2d(df_, 'mh_', 'Lz_', ('mh_', 'size'), mh_edges, Lz_edges, cmap=cmap)
#plt.axvline(0.45, ls=':', lw=0.5, c='m')
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
plt.ylabel(r"$L_z$ [$10^3\times$ kpc km s$^{-1}$]")

plt.subplot(1, 4, 2)
#cmap = sns.color_palette('rocket_r', as_cmap=True)
cmap = sns.cubehelix_palette(start=2, rot=0, dark=-5.0, light=1.0, as_cmap=True)
plot_hist2d(df_, 'mh_', 'eta_', ('mh_', 'size'), mh_edges, eta_edges, cmap=cmap)
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
plt.ylabel(r"$L_z/L_\mathrm{c}$")

plt.subplot(1, 4, 3)
#cmap = 'cubehelix_r'
#cmap = sns.color_palette('mako_r', as_cmap=True)
#cmap = sns.color_palette("Blues", as_cmap=True)
#cmap = sns.color_palette("ch:start=.1,rot=-.3,light=1.0", as_cmap=True)
cmap = sns.color_palette("ch:start=.1,rot=-.3,dark=-1.0,light=1.0", as_cmap=True)
plot_hist2d(df_, 'mh_', 'apo_', ('mh_', 'size'), mh_edges, apo_edges, cmap=cmap)
#plt.axvline(0.45, ls=':', lw=0.5, c='m')
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
plt.ylabel(r"$r_\mathrm{apo}$ [kpc]")
plt.yscale('log')

plt.subplot(1, 4, 4)
#cmap = 'gist_yarg'
cmap = sns.color_palette("ch:dark=-1.0,light=1.0", as_cmap=True)
plot_hist2d(df_, 'mh_', 'ecc_', ('mh_', 'size'), mh_edges, ecc_edges, cmap=cmap)
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
plt.ylabel(r"$e$")

plt.savefig("fig/data-kinem-mh.pdf")
if show_figures:  plt.show()
plt.close()

# %%



# %%
# 2-D histograms splitted by R

def plot_row(df_, axrow):

    ax_ = axrow[0]
    hist, lz_edges, mh_edges = np.histogram2d(df_['Lz']/1e3, df_['mh_xgboost'], range=((-2.0, 4.0), (-3.0, 0.5)), bins=(60, 90))
    ####hist_min = hist.min(axis=0)[np.newaxis,:]
    ####hist_max = hist.max(axis=0)[np.newaxis,:]
    ####hist = (hist - hist_min) / (hist_max - hist_min)
    hist /= hist.sum(axis=0)
    cmap = 'gist_heat_r'
    ax_.pcolormesh(mh_edges, lz_edges, hist, cmap=cmap, alpha=0.9, rasterized=True)
    ax_.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
    ax_.set_ylabel(r"$L_z$ [$10^3\times$ kpc km s$^{-1}$]")

    ax_ = axrow[1]
    hist, lzlc_edges, mh_edges = np.histogram2d(df_['Lz']/df_['Lc'], df_['mh_xgboost'], range=((-1.0, 1.0), (-3.0, 0.5)), bins=(60, 90))
    ####hist_min = hist.min(axis=0)[np.newaxis,:]
    ####hist_max = hist.max(axis=0)[np.newaxis,:]
    ####hist = (hist - hist_min) / (hist_max - hist_min)
    hist /= hist.sum(axis=0)
    cmap = sns.cubehelix_palette(start=2, rot=0, dark=-5.0, light=1.0, as_cmap=True)
    ax_.pcolormesh(mh_edges, lzlc_edges, hist, cmap=cmap, rasterized=True)
    ax_.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
    ax_.set_ylabel(r"$L_z/L_\mathrm{c}$")

    ax_ = axrow[2]
    hist, rapo_edges, mh_edges = np.histogram2d(df_['apo'], df_['mh_xgboost'], range=((0.0, 20.0), (-3.0, 0.5)), bins=(60, 90))
    ####hist_min = hist.min(axis=0)[np.newaxis,:]
    ####hist_max = hist.max(axis=0)[np.newaxis,:]
    ####hist = (hist - hist_min) / (hist_max - hist_min)
    hist /= hist.sum(axis=0)
    cmap = sns.color_palette("ch:start=.1,rot=-.3,dark=-1.0,light=1.0", as_cmap=True)
    ax_.pcolormesh(mh_edges, rapo_edges, hist, cmap=cmap, rasterized=True)
    ax_.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
    ax_.set_ylabel(r"$r_\mathrm{apo}$ [kpc]")

    ax_ = axrow[3]
    hist, rapo_edges, mh_edges = np.histogram2d(df_['ecc'], df_['mh_xgboost'], range=((0.0, 1.0), (-3.0, 0.5)), bins=(60, 90))
    ####hist_min = hist.min(axis=0)[np.newaxis,:]
    ####hist_max = hist.max(axis=0)[np.newaxis,:]
    ####hist = (hist - hist_min) / (hist_max - hist_min)
    hist /= hist.sum(axis=0)
    #cmap = 'gist_yarg'
    cmap = sns.color_palette("ch:dark=-1.0,light=1.0", as_cmap=True)
    ax_.pcolormesh(mh_edges, rapo_edges, hist, cmap=cmap, rasterized=True)
    ax_.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
    ax_.set_ylabel(r"eccentricity")


r_cut1, r_cut2 = 3.0, 4.0


plt.rc('font', size=5.0)
inch = 2.54  ## cm
width, height = 17/inch, 10/inch
#plt.figure(figsize=(width, height), layout='constrained')
_, ax = plt.subplots(3, 4, figsize=(width, height), layout='constrained')

mask = df['R'] < r_cut1
plot_row(df[mask], ax[0])
ax[0,0].annotate(r"$R < {}$ kpc".format(r_cut1), xy=(0.1, 0.85), xycoords='axes fraction')

mask = (r_cut1 <= df['R']) & (df['R'] < r_cut2)
plot_row(df[mask], ax[1])
ax[1,0].annotate(r"${} \leq R < {}$ kpc".format(r_cut1, r_cut2), xy=(0.1, 0.85), xycoords='axes fraction')

mask = r_cut2 <= df['R']
plot_row(df[mask], ax[2])
ax[2,0].annotate(r"${} \leq R$ kpc".format(r_cut2), xy=(0.1, 0.85), xycoords='axes fraction')

for ax_ in ax[-1]:
    ax_.set_xlabel(r"$[\mathrm{M}/\mathrm{H}]$")


plt.savefig("fig/data-kinem-mh-rsplit.pdf")
if show_figures:  plt.show()
plt.close()

# %%



# %%
# 2-D plots colored by apo

log10median = lambda x : np.median(np.log10(x))
axis=0
mask = df['apo'] < 10**1.5
#mask &= df['parallax_over_error'] > 10
df_ = df[mask]

plt.rc('font', size=5.0)
inch = 2.54  ## cm
#width, height = 17/inch, 5.69/inch
width, height = 17/inch, 3.5/inch
plt.figure(figsize=(width, height), layout='constrained')

cmap = 'jet'

plt.subplot(1, 4, 1)
cbar = plot_hist2d(df_, 'mh_', 'Lz_', ('apo', log10median), mh_edges, Lz_edges, cmap=cmap, colnorm=False, cb=True)
cbar.set_label(r"$r_\mathrm{apo}$ [kpc]")
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
plt.ylabel(r"$L_z$ [$10^3\times$ kpc km s$^{-1}$]")

plt.subplot(1, 4, 2)
plot_hist2d(df_, 'mh_', 'eta_', ('apo', log10median), mh_edges, eta_edges, cmap=cmap, colnorm=False, cb=True)
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
plt.ylabel(r"$L_z/L_\mathrm{c}$")

plt.subplot(1, 4, 3)
#cmap_ = sns.color_palette("ch:start=.1,rot=-.3,light=1.0", as_cmap=True)
cmap_ = sns.color_palette("ch:start=.1,rot=-.3,dark=-1.0,light=1.0", as_cmap=True)
plot_hist2d(df_, 'mh_', 'apo_', ('mh_', 'size'), mh_edges, apo_edges, cmap=cmap_)
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
plt.ylabel(r"$r_\mathrm{apo}$ [kpc]")
plt.yscale('log')

plt.subplot(1, 4, 4)
plot_hist2d(df_, 'mh_', 'ecc_', ('apo', log10median), mh_edges, ecc_edges, cmap=cmap, colnorm=False, cb=True)
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
plt.ylabel(r"$e$")

plt.savefig("fig/data-kinem-mh-logapocolor.pdf")
if show_figures:  plt.show()
plt.close()

# %%



# %%
# L_z/L_c vs. apo

"""
mask = (-2.0 < df['mh_xgboost']) & (df['mh_xgboost'] <= -1.5)
#mask &= df['parallax_over_error'] > 10
df_ = df[mask]

plt.figure()

plot_hist2d(df_, 'apo_', 'eta_', ('apo_', 'size'), apo_edges, eta_edges, cmap='gist_heat_r')
plt.xlabel(r"$r_\mathrm{apo}$ [kpc]")
plt.xscale('log')
plt.ylabel(r"$L_z/L_\mathrm{c}$")

#if show_figures:  plt.show()
plt.close()
"""

# %%



# %%
# Make the Gaussian Mixture Model decomposition on the L_z/L_c vs. apo plane

"""
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# Metallicity range to select
mh_lo_lim_, mh_hi_lim_ = -2.0, -1.5
#mh_lo_lim_, mh_hi_lim_ = -3.0, -2.0

mask = (mh_lo_lim_ < df['mh_xgboost']) & (df['mh_xgboost'] <= mh_hi_lim_) & (df['apo'] < 10**1.5)
#mask &= df['parallax_over_error'] > 10
mask &= (0.95 < np.log10(df['apo'])) & (np.log10(df['apo']) < 1.0) & (0.5 < df['eta']) & (df['eta'] < 0.8)
df_ = df[mask]
print(len(df_))

X = np.stack((np.log10(df_['apo']), df_['eta'])).T
print("shape(X) =", np.shape(X))

gms = {}
for n in [2, 3, 4]:
    gm = GaussianMixture(n_components=n, covariance_type='full')
    gm.fit(X)
    print("n =", n)
    print("  w:\t", gm.weights_)
    print("  m:\t", gm.means_[0])
    print("\t", gm.means_[1])
    print("  c:\t", gm.covariances_[0])
    print("\t", gm.covariances_[1])
    gms[n] = gm


def plot_ellipse(mean, cov, ax, **kwargs):
    # This example is stolen from the Matplotlib manual:
    # https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html
    #
    # The approach that is used to obtain the correct geometry is explained and proved here:
    # https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html

    pearson = cov[0,1]/np.sqrt(cov[0,0] * cov[1,1])

    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = mpl.patches.Ellipse((0, 0), width=2*ell_radius_x, height=2*ell_radius_y,
                                  lw=1.5, facecolor='none', **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    n_std = 1.0
    scale_x = n_std * np.sqrt(cov[0,0])
    scale_y = n_std * np.sqrt(cov[1,1])

    transf = mpl.transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


plt.rc('font', size=6.0)
inch = 2.54  ## cm
width, height = 8/inch, 6.5/inch
plt.figure(figsize=(width, height), layout='constrained')

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

ax = plt.gca()
ax_ = ax.twiny()

plt.title(f"{mh_lo_lim_} < [M/H] < {mh_hi_lim_}")
plot_hist2d(df_, 'apo_', 'eta_', ('apo_', 'size'), apo_edges, eta_edges, ax=ax, cmap='gist_heat_r', colnorm=False)
#for n in gms.keys():
for n in [2]:
    gm = gms[n]
    for i in range(n):
        #print(gm.means_[i], gm.covariances_[i])
        #plot_ellipse(gm.means_[i], gm.covariances_[i], ax_, edgecolor='green')
        pass
ax.set_xlabel(r"$r_\mathrm{apo}$ [kpc]")
ax.set_xscale('log')
ax_.set_xscale('linear')
ax_.set_xlim(np.log10(np.array(ax.get_xlim())))
ax_.set_xticks([])
ax.set_ylabel(r"$L_z/L_\mathrm{c}$")

plt.savefig('fig/data-eta-apo-imp.pdf')
#plt.savefig('fig/data-eta-apo-vmp.pdf')
if show_figures:  plt.show()
plt.close()
"""

# %%



# %%
# Kinematics diagram

plt.rc('font', size=5.0)
inch = 2.54  ## cm
width, height = 17/inch, 4.75/inch
plt.figure(figsize=(width, height), layout='constrained')

#
# eta vs [M/H]

plt.subplot(1, 3, 1)

df_ = df

rapo_gse_cut = 8.0


#
# eta vs [M/H]

ax = plt.gca()
#cmap = 'gist_yarg'
#cmap = 'gist_heat_r'
cmap = sns.cubehelix_palette(dark=-5.0, light=1.0, as_cmap=True)
plot_hist2d(df_, 'mh_', 'eta_', ('mh_', 'size'), mh_edges, eta_edges, ax=ax, cmap=cmap, maskzeros=False)
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
plt.xlim(-3.0, 0.5)
plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
plt.ylabel(r"$L_z/L_\mathrm{c}\;|\;[\mathrm{M}/\mathrm{H}]$")


#
# eta vs r_apo

plt.subplot(1, 3, 2)
ax = plt.gca()

mh_lo_lim_, mh_hi_lim_ = -2.0, -1.5
#mh_lo_lim_, mh_hi_lim_ = -3.0, -2.0
mask = (mh_lo_lim_ < df['mh_xgboost']) & (df['mh_xgboost'] <= mh_hi_lim_) & (df['apo'] < 10**1.5)
#mask &= df['parallax_over_error'] > 10
df_ = df[mask]

# All
cmap = 'gist_yarg'
plot_hist2d(df_, 'apo_', 'eta_', ('apo_', 'size'), apo_edges, eta_edges, ax=ax, cmap=cmap, colnorm=False)

# Aurora
mask = df_['apo'] < 10.0
df__ = df_[mask]
cmap = sns.color_palette("ch:start=0.1,rot=-0.3,dark=0.1,light=0.85", as_cmap=True)
plot_hist2d(df__, 'apo_', 'eta_', ('apo_', 'size'), apo_edges, eta_edges, ax=ax, cmap=cmap, colnorm=False)

# GSE
mask = (rapo_gse_cut < df_['apo']) & (df_['apo'] < 35.0) & (-0.25 < df_['eta']) & (df_['eta'] < 0.25)
df__ = df_[mask]
cmap = sns.cubehelix_palette(start=2, rot=0, dark=0.0, light=0.75, as_cmap=True)
plot_hist2d(df__, 'apo_', 'eta_', ('apo_', 'size'), apo_edges, eta_edges, ax=ax, cmap=cmap, alpha=0.7, colnorm=False)

# Captured
mask = (0.95 < np.log10(df_['apo'])) & (np.log10(df_['apo']) < 1.05) & (0.5 < df_['eta']) & (df_['eta'] < 0.8)
df__ = df_[mask]
cmap = sns.cubehelix_palette(dark=0.1, light=0.75, as_cmap=True)
plot_hist2d(df__, 'apo_', 'eta_', ('apo_', 'size'), apo_edges, eta_edges, ax=ax, cmap=cmap, alpha=0.7, colnorm=False)

ax.set_xlabel(r"$r_\mathrm{apo}$ [kpc]")
ax.set_xscale('log')
ax.set_ylabel(r"$L_z/L_\mathrm{c}$")


#
# E vs L_z

plt.subplot(1, 3, 3)
ax = plt.gca()

# All
cmap = 'gist_yarg'
plot_hist2d(df_, 'Lz_', 'E_', ('Lz_', 'size'), Lz_edges, E_edges, ax=ax, cmap=cmap, colnorm=False)

# Aurora
mask = df_['apo'] < 10.0
df__ = df_[mask]
cmap = sns.color_palette("ch:start=0.1,rot=-0.3,dark=0.1,light=0.85", as_cmap=True)
plot_hist2d(df__, 'Lz_', 'E_', ('Lz_', 'size'), Lz_edges, E_edges, ax=ax, cmap=cmap, colnorm=False)

# GSE
mask = (rapo_gse_cut < df_['apo']) & (df_['apo'] < 35.0) & (-0.25 < df_['eta']) & (df_['eta'] < 0.25)
df__ = df_[mask]
cmap = sns.cubehelix_palette(start=2, rot=0, dark=0.0, light=0.75, as_cmap=True)
plot_hist2d(df__, 'Lz_', 'E_', ('Lz_', 'size'), Lz_edges, E_edges, ax=ax, cmap=cmap, alpha=0.7, colnorm=False)

# Captured
mask = (0.95 < np.log10(df_['apo'])) & (np.log10(df_['apo']) < 1.05) & (0.5 < df_['eta']) & (df_['eta'] < 0.8)
df__ = df_[mask]
cmap = sns.cubehelix_palette(dark=0.1, light=0.75, as_cmap=True)
plot_hist2d(df__, 'Lz_', 'E_', ('Lz_', 'size'), Lz_edges, E_edges, ax=ax, cmap=cmap, alpha=0.7, colnorm=False)

ax.set_xlabel(r"$L_z$ [$10^3\times$ kpc km s$^{-1}$]")
plt.xlim(-2.0, 2.0)
ax.set_ylabel(r"$E$ [$10^5\times$ km$^2$ s$^{-2}$]")

# Plot the decision boundary to separate stars into accreted (high energy) and Aurora (in-situ, low energy populations)
# Belokurov, Kravtsov (2023) https://arxiv.org/abs/2306.00060

Lz = np.linspace(-2.0, 2.0, 200)

E = np.empty_like(Lz)
#
mask = Lz <= -0.58
E[mask] = -1.3
#
mask = (-0.58 < Lz) & (Lz < 0.58)
E[mask] = -1.4 + 0.3*Lz[mask]**2
#
mask = Lz > 0.58
E[mask] = -1.325 + 0.075*Lz[mask]**2

ax.plot(Lz, E, ':k')


plt.savefig('fig/data-kinem-all.pdf')
if show_figures:  plt.show()
plt.close()

# %%



# %%
# Kinematics diagram - 2

plt.rc('font', size=5.0)
inch = 2.54  ## cm
width, height = 12/inch, 5.5/inch
plt.figure(figsize=(width, height), layout='constrained')

mh_lo_lim_, mh_hi_lim_ = -2.0, -1.5
#mh_lo_lim_, mh_hi_lim_ = -3.0, -2.0
mask = (mh_lo_lim_ < df['mh_xgboost']) & (df['mh_xgboost'] <= mh_hi_lim_) & (df['apo'] < 10**1.5)
#mask &= df['parallax_over_error'] > 10
df_ = df[mask]

#
# eta vs r_apo

plt.subplot(1, 2, 1)
ax = plt.gca()
cmap = 'gist_yarg'
plot_hist2d(df_, 'apo_', 'eta_', ('apo_', 'size'), apo_edges, eta_edges, ax=ax, cmap=cmap, colnorm=False)
ax.set_xlabel(r"$r_\mathrm{apo}$ [kpc]")
ax.set_xscale('log')
ax.set_ylabel(r"$L_z/L_\mathrm{c}$")

#
# E vs L_z

plt.subplot(1, 2, 2)
ax = plt.gca()
cmap = 'gist_yarg'
plot_hist2d(df_, 'Lz_', 'E_', ('Lz_', 'size'), Lz_edges, E_edges, ax=ax, cmap=cmap, colnorm=False)
ax.set_xlabel(r"$L_z$ [$10^3\times$ kpc km s$^{-1}$]")
plt.xlim(-2.0, 2.0)
ax.set_ylabel(r"$E$ [$10^5\times$ km$^2$ s$^{-2}$]")

plt.savefig('fig/data-kinem-all-2.pdf')
if show_figures:  plt.show()
plt.close()

# %%



# %%
# Select the weird stars

#mask = (df['mh_xgboost'] > 0.4) & (df['Lz'] < 1e3)
mask = (df['R'] < 3)
df_ = df[mask]


plt.rc('font', size=8.0)
inch = 2.54  ## cm
width, height = 12/inch, 5/inch
plt.figure(figsize=(width, height), layout='constrained')

plt.subplot(1, 2, 1)
tmp = df_.groupby('mh_', observed=True)['vR'].std()
plt.stairs(tmp, mh_edges, baseline=None)
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
plt.ylabel(r"$\sigma_{v_R}$ [km s$^{-1}$]")

plt.subplot(1, 2, 2)
tmp = df_.groupby('mh_', observed=True)['vz'].std()
plt.stairs(tmp, mh_edges, baseline=None)
plt.xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
plt.ylabel(r"$\sigma_{v_z}$ [km s$^{-1}$]")

plt.savefig('fig/data-kinem-sigmas.pdf')
if show_figures:  plt.show()
plt.close()

# %%

sys.exit(0)



# %%
# Sky projection

mh_lo_lim_, mh_hi_lim_ = -2.0, -1.5
#mh_lo_lim_, mh_hi_lim_ = -3.0, -2.0

mask = (mh_lo_lim_ < df['mh_xgboost']) & (df['mh_xgboost'] <= mh_hi_lim_) & (df['apo'] < 10**1.5)
#mask &= df['parallax_over_error'] > 10
mask &= ~df['source_id'].isin(gc['source_id'])
mask &= (0.95 < np.log10(df['apo'])) & (np.log10(df['apo']) < 1.05) & (0.5 < df['eta']) & (df['eta'] < 0.8)
df_ = df[mask]
print(len(df_))

# %%

l = (df_['l'] + 180) % 360 - 180

plt.figure()
#plt.scatter(df_['l'], df_['b'], s=0.5)
plt.hist2d(l, df_['b'], bins=75)
plt.colorbar()
plt.show()


#
# Distances
plt.figure()
plt.hist(df_['dist'], bins=50)
plt.show()

len(df_)

# %%

plt.figure()
#plt.hist(df_['ecc'], bins=50)
#plt.hist(df_['eta'], bins=50)
plt.hist(df_['vphi'], bins=50)
plt.show()

df_['vz'].std()

# %%
