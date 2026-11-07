# %%

import sys
#sys.path.append('..')
import importlib

import pickle
from tqdm import tqdm

import numpy as np
import scipy as sp
from scipy.special import erf
import pandas as pd

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord

import matplotlib as mpl
import matplotlib.pyplot as plt

import config
from domain import Domain
import utils

# %%
# Load everything

#
# Load domain

print("\n* Load domain")

dom = Domain().load(config.data_path + f"/domain_hpx{config.hpxorder}.npz")
D = dom.D
D_edges = dom.D_edges
G = dom.G
G_edges = dom.G_edges
dG = dom.dG
dVol = dom.dVol
Omega = dom.Omega

#
# Load AG map

print("\n* Load AG map")

# New
res = np.load(config.data_path + '/extinctions.npz')
AG_map0 = res['ag_map']
# Old
#res = np.load('s4_ag_map.npz')
#AG_map0 = res['AG_map']

print("  shape(AG_map0) =", AG_map0.shape)

# Downscale the map if necessary
print("  Downscale the map if necessary")
"""
if dom.hpxorder < 7:
    tmp = pd.DataFrame({'hpx7': np.arange(len(AG_map0)), 'ag': AG_map0})
    tmp['hpx'] = tmp['hpx7'] // 4**(7-dom.hpxorder)
    AG_map = tmp.groupby('hpx', observed=False)['ag'].median().values
else:
    AG_map = AG_map0
"""
AG_map = np.zeros(dom.npix)
if dom.hpxorder < 7:
    for i0 in range(AG_map0.shape[0]):
        i = i0 // 4**(7-dom.hpxorder)
        AG_map[i] += AG_map0[i0] / 4**(7-dom.hpxorder)
print("  shape(AG_map) =", AG_map.shape)
del(res)

#
# Load model for parallax errors

print("\n* Load model for parallax errors")

with open(config.data_path + '/logplxerr_nvis_g.pkl', 'rb') as f:
    nvis_bins = pickle.load(f)
    plxerr_nvis_g = pickle.load(f)

#
# Load aggregated statistics on parallax errors

print("\n* Load aggregated statistics on parallax errors")

stats0 = pd.read_hdf(config.data_path + '/stats.hdf', key='df')
print("Nans:")
print(stats0.isna().sum())

mask = stats0['g_bin'] >= 0
stats = stats0[mask].copy()
stats.dropna(inplace=True)

# Reduce the HEALPix resolution to the required level
stats['hpx'] = stats['hpx7'] // 4**(7-dom.hpxorder)

# Evaluate a HEALPix-median for 'visibility_periods_used'
peh = stats.groupby('hpx').agg(nvis=('visibility_periods_used', 'median')).reset_index()
peh['nvis_bin'] = pd.cut(peh['nvis'], bins=nvis_bins).astype(str)

# Sort by a 'hpx' values
peh.sort_values('hpx', ascending=True, inplace=True)

del(stats0)
del(stats)

#
# Load subsample SF of XP/GS

print("\n* Load subsample SF of XP/GS")

fpath = config.data_path + f"/selfun_xp_hpx{dom.hpxorder}.npz"
res = np.load(fpath)
#prb_xp = res['prb_xp']
prb_xp = res['prb_xp_smo']
print("shape(prb_xp) =", prb_xp.shape)
del(res)

# %%
# Evaluate transfer function and auxiliaries

# Fine grid along `G` axis
# Dims: GG
G_stride = 10
GG_edges = np.linspace(G_edges[0], G_edges[-1], G_stride*len(G)+1)
GG = 0.5*(GG_edges[:-1] + GG_edges[1:])
dGG = GG_edges[1:] - GG_edges[:-1]

# Dims: hpx, G, D'
MG = np.nan * np.zeros((len(dom.ipix), len(G), len(D)))
# Dims: hpx, G, D_edges'
MG_edges = np.nan * np.zeros((len(dom.ipix), len(G), len(D_edges)))

# Dims: hpx, D, G
SF = np.nan * np.zeros((len(dom.ipix), len(D), len(G)))

# Dims: hpx, D, G, D'
TF0 = np.nan * np.zeros((len(dom.ipix), len(D), len(G), len(D)))
TF = np.nan * np.zeros((len(dom.ipix), len(D), len(G), len(D)))

# Selection cuts
G_lim = 16.0
q = 4.0

# Smearing or parallax noise kernel
print("\nPass 1")
for k, hpx in tqdm(enumerate(dom.ipix), total=len(dom.ipix)):
    # Dims: GG, D'
    MG_ = GG[:,None] - AG_map[hpx] - 5.0*np.log10(D[None,:]) - 10.0
    # Dims: GG, D_edges'
    MG_edges_ = GG[:,None] - AG_map[hpx] - 5.0*np.log10(D_edges[None,:]) - 10.0
    # Dims: G, D'
    MG[hpx] = MG_.reshape((len(G),G_stride,len(D))).mean(axis=1)
    # Dims: G, D_edges'
    MG_edges[hpx] = MG_edges_.reshape((len(G),G_stride,len(D_edges))).mean(axis=1)

    # Parallax error
    nvis_bin = peh.iloc[hpx]['nvis_bin']
    # Dims: G
    sigma_plx = 10**np.interp(G, plxerr_nvis_g[nvis_bin]['G'], plxerr_nvis_g[nvis_bin]['logsigma_plx'])
    ## Model by Gaia
    #G_ = G.copy()
    #G_[G < 13] = 13
    #z = 10**(0.4*(G_-15))
    #sigma_plx = 1e-3*np.sqrt(40 + 800*z + 30*z**2)

    # Dims: D, G, D'
    TF0[hpx] = 0.5*erf((1/D_edges[:-1,None,None] - 1.0/D[None,None,:]) / (np.sqrt(2)*sigma_plx[None,:,None])) \
             - 0.5*erf((1/D_edges[1:,None,None]  - 1.0/D[None,None,:]) / (np.sqrt(2)*sigma_plx[None,:,None]))

    isnan = np.any(np.isnan(TF0[hpx]))
    if isnan:
        print(f"hpx={hpx}: NaN")

# Normalize the noise kernel over the D space
print("\nPass 2")
for k, hpx in tqdm(enumerate(dom.ipix), total=len(dom.ipix)):
    # Summarize over the observed distances
    # Dims: G, D'
    norm = TF0[hpx].sum(axis=0)
    # Dims: D, G, D'
    TF0[hpx] = np.einsum('jkl,kl->jkl', TF0[hpx], 1./norm)

print(np.allclose(TF0.sum(axis=1), 1.0))

# Apply selections
print("\nPass 3")
for k, hpx in tqdm(enumerate(dom.ipix), total=len(dom.ipix)):

    # Magnitude selection
    # Dims: G
    SF_mag_ = G <= G_lim
    # XP selection
    # Dims: G
    SF_ssf_interp = sp.interpolate.interp1d(G, prb_xp[hpx], bounds_error=False, fill_value=(0.0, 0.0))
    SF_ssf = SF_ssf_interp(G)

    # Parallax selection
    # Parallax error
    nvis_bin = peh.iloc[hpx]['nvis_bin']
    # Dims: G
    sigma_plx = 10**np.interp(G, plxerr_nvis_g[nvis_bin]['G'], plxerr_nvis_g[nvis_bin]['logsigma_plx'])
    ## Model by Gaia
    #G_ = G.copy()
    #G_[G < 13] = 13
    #z = 10**(0.4*(G_-15))
    #sigma_plx = 1e-3*np.sqrt(40 + 800*z + 30*z**2)
    # Dims: D, G
    SF_plx = 1/D[:,None] > q*sigma_plx[None,:]

    # Dims: D, G
    SF[hpx] = np.einsum('k,jk->jk', (SF_mag_.astype(float) * SF_ssf), SF_plx)

    # Multiply by selection functions and by a volume element in the true distances (for the model density)
    # Dims: D, G, D'
    TF[hpx] = np.einsum('jk,jkl->jkl', SF[hpx], TF0[hpx])

    isnan = np.any(np.isnan(TF[hpx]))
    if isnan:
        print(f"hpx={hpx}: NaN")

# %%



# %%
# Plot

plt.rc('font', size=6.0)
inch = 2.54  ## cm
width, height = 17/inch, 15/inch
plt.figure(figsize=(width, height), layout='constrained')
plt.suptitle("CF density on sky position")

l = [ 0, 30]
b = [-5,  7]
co = SkyCoord(l, b, frame='galactic', unit='deg')
hp = HEALPix(nside=dom.nside, order='nested', frame='icrs')
ipix_ = hp.skycoord_to_healpix(co)
print("ipix_:", ipix_)

for i, hpx in enumerate(ipix_):
    plt.subplot(2, 2, 1+i)
    plt.title(f"(l,b) = ({co[i].l.deg}, {co[i].b.deg})")
    l = 18
    ih = np.argwhere(dom.ipix == hpx)[0][0]
    im = plt.pcolormesh(D_edges, G_edges, TF[ih][...,l].T, cmap='inferno', norm='log', rasterized=True, label=f"D' = {D[l]:.1f}")
    plt.legend()
    plt.colorbar(im, ax=plt.gca())
    plt.xlabel("$D$ [kpc]")
    plt.ylabel("$G$ [mag]")

    plt.subplot(2, 2, 3+i)
    l = 25
    ih = np.argwhere(dom.ipix == hpx)[0][0]
    im = plt.pcolormesh(D_edges, G_edges, TF[ih][...,l].T, cmap='inferno', norm='log', rasterized=True, label=f"D' = {D[l]:.1f}")
    plt.legend()
    plt.colorbar(im, ax=plt.gca())
    plt.xlabel("$D$ [kpc]")
    plt.ylabel("$G$ [mag]")

plt.savefig('fig/cf_1.pdf')
#plt.show()
plt.close()

# %%



# %%
# Diagnostics

import phys_models
importlib.reload(phys_models)

eta = [0.25, 0.042/0.25, 0.005/0.042]
# Dims: hpx, G, D'
Phi = phys_models.lumfun_linear(MG, eta)
dPsi_ = np.einsum('ikl,k->ikl', Phi, dG)
norm = dPsi_.sum(axis=1)
mask = norm > 0.0
norm_inv = np.zeros_like(norm)
norm_inv[mask] = 1./norm[mask]
dPsi = np.einsum('ikl,il->ikl', dPsi_, norm_inv)

# Dims: hpx, D, D'
CF_lbDD = np.einsum('ijkl,ikl->ijl', TF, dPsi)
print("shape(CF_lbDD) =", CF_lbDD.shape)

# %%

plt.rc('font', size=6.0)
inch = 2.54  ## cm
width, height = 16/inch, 25/inch
plt.figure(figsize=(width, height), layout='constrained')
plt.suptitle("CF density on sky position")

l = [ 0, 30]
b = [-5,  7]
co = SkyCoord(l, b, frame='galactic', unit='deg')
hp = HEALPix(nside=dom.nside, order='nested', frame='icrs')
ipix_ = hp.skycoord_to_healpix(co)
print("ipix_:", ipix_)

d = 18
print(f"D[{d}] = {D[d]:g} [kpc]")
g = 16
print(f"G[{g}] = {G[g]:g} [mag]")
for i, hpx in enumerate(ipix_):
    ih = np.argwhere(dom.ipix == hpx)[0][0]

    plt.subplot(4, 2, 1+0+i)
    plt.plot(MG[ih][:,d], dPsi[ih][:,d], label=f"D = {D[d]:.3g} kpc")
    plt.xlabel("$M$ [mag]")

    plt.subplot(4, 2, 1+2+i)
    plt.title(f"(l,b) = ({co[i].l.deg}, {co[i].b.deg})")
    im = plt.pcolormesh(D, D, CF_lbDD[ih], norm='log', cmap='inferno', rasterized=True)
    plt.colorbar(im, ax=plt.gca())
    plt.axvline(D[d], ls=':', c='k')
    plt.axhline(D[d], ls=':', c='k')
    plt.xlabel("$D'$ [kpc]")
    plt.ylabel("$D$ [kpc]")

    plt.subplot(4, 2, 1+4+i)
    plt.plot(D, CF_lbDD[ih,d,:], '-', label=f"D = {D[d]:.3g} kpc")
    plt.axvline(D[d], ls=':', c='k')
    plt.legend(frameon=False)
    plt.xlabel("$D'$ [kpc]")

    plt.subplot(4, 2, 1+6+i)
    plt.plot(D, CF_lbDD[ih,:,d], '-', label=f"D' = {D[d]:.3g} kpc")
    #plt.plot(G, dPsi[ih,d,:], '-')
    #plt.plot(D, TF[ih,:,g,d], '-')
    plt.axvline(D[d], ls=':', c='k')
    plt.legend(frameon=False)
    plt.xlabel("$D$ [kpc]")

plt.savefig('fig/cf_2.pdf')
plt.show()
plt.close()
#sys.exit(0)

# %%
# Save

print("\n* Save transfer function and auxiliaries")
print("isnan(SF) =", np.any(np.isnan(SF)))
print("isnan(TF) =", np.any(np.isnan(TF)))

np.savez_compressed(config.data_path + f"/transfun_hpx{dom.hpxorder}.npz",
                    MG=MG, MG_edges=MG_edges, SF=SF, TF=TF)

# %%
