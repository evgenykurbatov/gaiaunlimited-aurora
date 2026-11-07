# %%

import sys
import pathlib
from tqdm import tqdm

import numpy as np
import scipy as sp
import pandas as pd

import astropy_healpix

import matplotlib as mpl
import matplotlib.pyplot as plt

import config

# %%
# Configure plots

fig_path = 'fig'
fig_dpi = 600

def fig_init(dims=(8.0, 7.0), shape=(1, 1), font_size=7.0, layout='constrained', **kwargs):
    plt.rc('font', size=font_size)
    cm_per_in = 2.54
    fig, ax = plt.subplots(shape[1], shape[0],
                           figsize=(shape[0]*dims[0]/cm_per_in, shape[1]*dims[1]/cm_per_in),
                           layout=layout, **kwargs)
    return fig, ax

# %%
# Load GC catalogue

print("\n* Load GC catalogue")

gc = pd.read_hdf('baum+bel-gc.hdf')
print("  records:", len(gc))
print("   memory: %.1f MiB" % (gc.memory_usage().sum()/(1024**2)))
print(list(gc.columns))
gc

# %%

sorted(set(gc['name']))

# %%
# Load GC stars catalogue

print("\n* Load GC stars catalogue")

columns = [ 'source_id', 'x', 'y', 'plx', 'plxe', 'g_mag', 'bp_rp', 'memberprob', 'vaname' ]
gcstars0 = pd.read_hdf('vasiliev2021-gcstars.hdf', columns=columns)
print("  records:", len(gcstars0))
print("   memory: %.1f MiB" % (gcstars0.memory_usage().sum()/(1024**2)))
print(list(gcstars0.columns))
gcstars0

# %%
# Load RGB star catalogue by Andrae et al. (2023)
#
# Paper:
#   Andrae R, Rix H-W, Chandra V. Robust Data-driven Metallicities for 175 Million Stars from Gaia XP Spectra (2023)
#   [ADS](https://ui.adsabs.harvard.edu/abs/2023ApJS..267....8A)
#
# Data:
#   [Zenodo](https://zenodo.org/records/7945154)

print("\n* Load RGB star catalogue by Andrae et al. (2023)")

columns = [ 'source_id', 'phot_g_mean_mag', 'teff_xgboost', 'logg_xgboost', 'mh_xgboost' ]
an = pd.read_hdf(config.data_path + '/gaia_xp_xgboost_v2.1.data/table_2_catwise.hdf', columns=columns)
print("  records:", len(an))
print("   memory: %.1f MiB" % (an.memory_usage().sum()/(1024**2)))
print(list(an.columns))
an

# %%

plt.figure()

plt.subplot(1, 2, 1)
plt.title("GCs")
plt.hist(an['phot_g_mean_mag'], histtype='step', label='RGB stars')
plt.hist(gcstars0['g_mag'],     histtype='step', label='GC stars')
plt.legend()
plt.xlabel("G [mag]")
plt.yscale('log')
plt.ylim(ymin=0.3)

plt.subplot(1, 2, 2)
plt.title("RGB")
plt.hist(an['logg_xgboost'], histtype='step', label='RGB stars')
plt.legend()
plt.xlabel("log(g)")

plt.show()
plt.close()

# %%



# %%

hpxorder = 7
hpxbase = 2**35 * 4**(12 - hpxorder)
nside = 2**hpxorder
npix = 12 * 4**hpxorder
ipix = np.arange(npix)
print(f"hpxorder={hpxorder} -> (nside={nside}, npix={npix})")

hpxangres = np.sqrt(astropy_healpix.nside_to_pixel_area(nside)).value * 180/np.pi * 60
print(f"hpxangres = {hpxangres:.3g} arcmin")

# %%
# Load subsample SF of XP/GS
#

print("\n* Load subsample SF of XP/GS")

fpath = f"sf_xp_hpx{hpxorder}_g.npz"
res = np.load(fpath)
G_edgs = res['G_']
#prb_xp = res['prb_xp']
prb_xp = res['prb_xp_smo']
print("shape(prb_xp) =", prb_xp.shape)

del(res)

#
# Load subsample SF of RGB/XP

print("\n* Load subsample SF of RGB/XP")

fpath = f"sf_rgb_hpx{hpxorder}_g.npz"
res = np.load(fpath)
G_edgs = res['G_']
#prb_rgb = res['prb_rgb']
prb_rgb = res['prb_rgb_smo']
print("shape(prb_rgb) =", prb_rgb.shape)

del(res)

# %%
# Load extinction maps
#

print("\n* Load extinction maps")

res = np.load(config.data_path + '/2023 Aurora/' + 's4_am_map.npz')
AG_map0  = res['AG_map']
ABP_map0 = res['ABP_map']
ARP_map0 = res['ARP_map']
print("shape(AG_map0) =", AG_map0.shape)

# Downsample the map if necessary
AG_map  = np.zeros(npix)
ABP_map = np.zeros(npix)
ARP_map = np.zeros(npix)
if hpxorder < 7:
    for i0 in range(AG_map0.shape[0]):
        i = i0 // 4**(7-hpxorder)
        AG_map[i]  += AG_map0[i0]  / 4**(7-hpxorder)
        ABP_map[i] += ABP_map0[i0] / 4**(7-hpxorder)
        ARP_map[i] += ARP_map0[i0] / 4**(7-hpxorder)
else:
    AG_map  = AG_map0
    ABP_map = ABP_map0
    ARP_map = ARP_map0
print("shape(AG_map) =", AG_map.shape)

am = pd.DataFrame({'hpx': ipix, 'ag_mag': AG_map, 'abp_mag': ABP_map, 'arp_mag': ARP_map})

del(res)

# %%



# %%
# Leave only high-probability members

mask = gcstars0['memberprob'] > 0.9
gcstars = gcstars0[mask].copy()
print("gcstars records:", len(gcstars))
print("   memory: %.1f MiB" % (gcstars.memory_usage().sum()/(1024**2)))

# %%
# Add some global GC parameters to GS stars

gcstars = gcstars.merge(gc[['D', 'r_hm', 'feh', 'vaname']], on='vaname', how='inner')
#gcstars

# %%
# Merge the GC stars with the extinctions

gcstars['hpx'] = gcstars['source_id'] // hpxbase
gcstars = gcstars.merge(am, on='hpx')

# %%
# Estimate magnitudes

gcstars['mg_mag'] = gcstars['g_mag'] - gcstars['ag_mag'] - 5.*np.log10(gcstars['D']) - 10.
gcstars['cg_mag'] = gcstars['bp_rp'] - (gcstars['abp_mag'] - gcstars['arp_mag'])
# https://gea.esac.esa.int/archive/documentation/GEDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_
gcstars['v_mag'] = gcstars['g_mag'] - ( -0.02704 + 0.01424*gcstars['cg_mag'] - 0.2156*gcstars['cg_mag']**2 + 0.01426*gcstars['cg_mag']**3 )

# %%
# Simulate Andrae selection on G and parallax quality

g_cut  = 16.0
plx_quality_cut = 4.0

gcstars['cut_andrae'] = (gcstars['g_mag'] < g_cut) & (gcstars['plx'] > plx_quality_cut*gcstars['plxe'])
print("# cut_andrae:", gcstars['cut_andrae'].sum())

# %%
# Select RGB stars by magnitude-color cut

cg_cut = 0.75
mg_cut = 3.0
#mg_cut = 1.5

gcstars['cut_rgb'] = (gcstars['cg_mag'] > cg_cut) & (gcstars['mg_mag'] < mg_cut)
print("# cut_rgb:", gcstars['cut_rgb'].sum())

# %%
# Avoid a dense core inside the half-mass radius

# Radial distance from the center of the GC [pc]
gcstars['r'] = gcstars['D']*1e3 * np.sqrt(gcstars['x']**2 + gcstars['y']**2) * np.pi/180
gcstars['cut_r'] = gcstars['r_hm'] < gcstars['r']

# %%
# Remove NaNs

gcstars.replace([np.inf, -np.inf], np.nan, inplace=True)
print("NaNs in gcstars:", gcstars.isna().sum().sum())
gcstars.isna().sum()
gcstars.dropna(inplace=True)

# %%



# %%
# Test the RGB and Andrae selection on a certain GC

vaname = 'NGC_6397'
#vaname = 'NGC_6266_M_62'  # 6.03 kpc
#vaname = 'NGC_7099_M_30'  # 8.46 kpc

mask = gcstars['vaname'] == vaname
print(f"# RGB stars in {vaname}:", len(gcstars[mask]))
mask &= gcstars['cut_rgb']
mask &= gcstars['cut_andrae']
print(f"# Cut RGB stars in {vaname}:", len(gcstars[mask]))

# %%
# Plot some distributions for stars of a certain GC

vaname = 'NGC_6397'
#vaname = 'NGC_6266_M_62'  # 6.03 kpc
#vaname = 'NGC_7099_M_30'  # 8.46 kpc

mask = gcstars['vaname'] == vaname
mask &= gcstars['cut_rgb'] & gcstars['cut_r']
#mask &= gcstars['cut_andrae']
gcstars_ = gcstars[mask]
print(f"# Cut RGB stars in {vaname}:", len(gcstars_))


_, ax = fig_init(shape=(2, 1), dims=(8.0, 6.5))
plt.suptitle(vaname)

ax_ = ax[0]
ax_.plot(gcstars_['x'], gcstars_['y'], 'o', ms=0.5, c='grey')
ax_.set_xlabel("x [pc]")
ax_.set_ylabel("y [pc]")

ax_ = ax[1]
_, _, _, im = ax_.hist2d(gcstars_['cg_mag'], gcstars_['mg_mag'], vmin=1, bins=100, norm='log', cmap='inferno')
plt.colorbar(im, ax=ax_)
ax_.set_xlabel(r"$\mathrm{BP} - \mathrm{RP}$ (corr.) [mag]")
ax_.set_ylabel(r"$M_\mathrm{G}$ [mag]")
ax_.yaxis.set_inverted(True)

plt.savefig(fig_path + f"/{vaname}.png", dpi=fig_dpi)
plt.show()
plt.close()

# %%




# %%
# Make the subsample selection function

def selfun(hpx, g_mag):
    G = 0.5*(G_edgs[:-1] + G_edgs[1:])
    #SF_interp = sp.interpolate.interp1d(G, prb_xp[int(hpx)] * prb_rgb[int(hpx)], bounds_error=False, fill_value=(0.0, 0.0))
    SF_interp = sp.interpolate.interp1d(G, prb_xp[int(hpx)], bounds_error=False, fill_value=(0.0, 0.0))
    return SF_interp(g_mag)

# %%
# Evaluate the subsample selection function

gcstars['selfun'] = gcstars.apply(lambda row: selfun(row['hpx'], row['g_mag']), axis=1).astype('float')
gcstars['selfun'].isna().sum()

# %%

gcstars['selfun'].describe()

# %%
# Plot HR diagram for the whole GC star [outer] population

_, ax = fig_init(shape=(2, 1), dims=(10.0, 8.0))
plt.suptitle("Whole GCs' stellar population")

ax_ = ax[0]
ax_.set_title("Not weighted")
_, _, _, im = ax_.hist2d(gcstars['cg_mag'], gcstars['mg_mag'], weights=None, vmin=1, bins=200, norm='log', cmap='inferno')
plt.colorbar(im, ax=ax_)
ax_.set_xlabel(r"$\mathrm{BP} - \mathrm{RP}$ (corr.) [mag]")
ax_.set_ylabel(r"$M_\mathrm{G}$ [mag]")
ax_.yaxis.set_inverted(True)

ax_ = ax[1]
ax_.set_title("Weighted by SSF")
_, _, _, im = ax_.hist2d(gcstars['cg_mag'], gcstars['mg_mag'], weights=gcstars['selfun'], vmin=1, bins=200, norm='log', cmap='inferno')
plt.colorbar(im, ax=ax_)
ax_.set_xlabel(r"$\mathrm{BP} - \mathrm{RP}$ (corr.) [mag]")
ax_.set_ylabel(r"$M_\mathrm{G}$ [mag]")
ax_.yaxis.set_inverted(True)

plt.savefig(fig_path + '/gcstars-hr.png', dpi=fig_dpi)
plt.show()
plt.close()

# %%



# %%
# Select GC stars after the final cuts by colors and absolute magnitudes (cut_rgb),
# apparent magnitudes and parallax quality (cut_andrae), and distances from GC centre (cut_r)

mask = gcstars['cut_rgb'] & gcstars['cut_r']
mask &= gcstars['cut_andrae']
gccut = gcstars[mask].copy()
print("# total cut GC stars:", len(gccut))

# %%
# Number of stars in the particular GC after cuts

vaname = 'NGC_6397'
#vaname = 'NGC_6266_M_62'  # 6.03 kpc
#vaname = 'NGC_7099_M_30'  # 8.46 kpc
print(f"# total cut RGB stars in {vaname}:", len(gccut[gccut['vaname'] == vaname]))

gccut[gccut['vaname'] == vaname]['ag_mag'].describe()

# %%

gccut['g_mag'].describe()

# %%



# %%
# Cross-match the Andrae stars with _selected_ GC stars

angc = an.merge(gccut, on='source_id', how='inner')
print("angc records:", len(angc))
print("   memory: %.1f MiB" % (angc.memory_usage().sum()/(1024**2)))
print(angc.columns.tolist())

# %%



# %%

_, ax = fig_init(shape=(2, 1), dims=(4.0, 3.9), font_size=5.0, layout='tight')
#_, ax = fig_init(shape=(2, 1), dims=(4.0, 4.5), font_size=5.0)

#mpl.style.use('tableau-colorblind10')
plt.style.use('seaborn-v0_8-colorblind')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

alpha = 0.75
linewidth = 2.0
markersize = 4.0

ax_ = ax[0]
#
mask_ = (2.9 < angc['logg_xgboost']) & (angc['logg_xgboost'] < 3.1)
ax_.hist(angc[mask_]['mg_mag'], bins=10, color='k', alpha=0.3, density=True)
#
hgc, edges = np.histogram(gccut['mg_mag'], bins=20)
hgcw, _ = np.histogram(gccut['mg_mag'], bins=20, weights=gccut['selfun'])
han, _ = np.histogram(angc['mg_mag'], bins=edges)
ax_.stairs(han/hgcw, edges, linewidth=linewidth, color=colors[0], alpha=alpha, label="Corrected")
ax_.stairs(han/hgc, edges, linewidth=linewidth, color=colors[2], alpha=alpha, label="Uncorrected")
#
ax_.legend(handlelength=0.7, frameon=False)
ax_.set_xlabel(r"$M_\mathrm{G}$ [mag]")
ax_.set_ylabel("rate")

ax_ = ax[1]
ax_.hist2d(angc['logg_xgboost'], angc['mg_mag'], bins=50, norm='log', cmap='inferno')
ax_.axvline(3.0, ls=':', c='darkgray')
ax_.set_xlabel(r"$\log g$")
ax_.set_xlim(0.0, 3.5)
ax_.xaxis.set_inverted(True)
ax_.set_ylabel(r"$M_\mathrm{G}$ [mag]")
ax_.yaxis.set_inverted(True)

#plt.savefig(fig_path + '/gc-andrae-rgb-rate-1.png', dpi=600)
plt.savefig(fig_path + '/gc-andrae-rgb-rate-1.pdf')
plt.show()
plt.close()

# %%



# %%

mg_edges = np.linspace(-6.0, 4.0, 11)
g_edges  = np.linspace(6.0, 17.0, 11)

vaname = 'NGC_6397'
#vaname = 'NGC_6266_M_62'  # 6.03 kpc
#vaname = 'NGC_7099_M_30'  # 8.46 kpc

mask = gccut['vaname'] == vaname
gccut_ = gccut[mask]

mask = np.full(len(angc), True)
#mask &= angc['logg_xgboost'] < 2.4
print("angc: mask on logg_xgboost:", (~mask).sum()/len(angc))
angc_ = angc[mask]

mask = angc_['vaname'] == vaname
angc__ = angc_[mask].copy()

print("mg_cut =", mg_cut)
print("cg_cut =", cg_cut)

print(f"{vaname}: rate   =", len(angc__)/len(gccut_))
print(f"{vaname}: rate_w =", len(angc__)/len(gccut_)/gccut_['selfun'].mean())


_, ax = fig_init(shape=(4, 2), dims=(8.0, 8.0))
alpha = 0.75

# One GC

ax_ = ax[0,0]
ax_.set_title(vaname)
ax_.plot(gccut_['cg_mag'], gccut_['mg_mag'], ls='', marker='o', c='grey', alpha=alpha, label="GC RGB")
ax_.plot(angc__['cg_mag'], angc__['mg_mag'], ls='', marker='.', c='red', alpha=alpha, label="Andrae x GC")
ax_.axhline(mg_cut, ls=':', c='grey')
ax_.axvline(cg_cut, ls=':', c='grey')
ax_.legend()
ax_.set_xlabel(r"$\mathrm{BP} - \mathrm{RP}$ (corr.) [mag]")
ax_.set_ylabel(r"$M_\mathrm{G}$ [mag]")
ax_.yaxis.set_inverted(True)

ax_ = ax[0,1]
ax_.set_title(vaname)
linewidth = 2.0
ax_.hist(gccut_['mg_mag'], bins=mg_edges,
         histtype='step', linewidth=linewidth, color='grey', alpha=alpha, label="GC RGB")
ax_.hist(gccut_['mg_mag'], bins=mg_edges, weights=gccut_['selfun'],
         histtype='step', linewidth=linewidth, linestyle='dashed', color='grey', alpha=alpha, label="GC RGB + SSF")
ax_.hist(angc__['mg_mag'], bins=mg_edges,
         histtype='step', linewidth=linewidth, color='red', alpha=alpha, label="Andrae X GC")
ax_.legend()
ax_.set_xlabel(r"$M_\mathrm{G}$ [mag]")

ax_ = ax[0,2]
ax_.set_title(vaname)
linewidth = 2.0
ax_.hist(gccut_['g_mag'], bins=g_edges,
         histtype='step', linewidth=linewidth, color='grey', alpha=alpha, label="GC RGB")
ax_.hist(gccut_['g_mag'], bins=g_edges, weights=gccut_['selfun'],
         histtype='step', linewidth=linewidth, linestyle='dashed', color='grey', alpha=alpha, label="GC RGB + SSF")
ax_.hist(angc__['g_mag'], bins=g_edges,
         histtype='step', linewidth=linewidth, color='red', alpha=alpha, label="Andrae X GC")
ax_.legend()
ax_.set_xlabel(r"$\mathrm{G}$ [mag]")

ax_ = ax[0,3]
ax_.set_title(vaname)
ax_.plot(angc__['teff_xgboost'], angc__['logg_xgboost'], ls='', marker='.', c='red', alpha=alpha, label="Andrae x GC")
ax_.legend()
ax_.set_xlabel(r"teff_xgboost")
ax_.xaxis.set_inverted(True)
ax_.set_ylabel(r"logg_xgboost")
ax_.yaxis.set_inverted(True)

# All stars

ax_ = ax[1,0]
ax_.set_title("Selected RGB stars")
ax_.plot(gccut['cg_mag'], gccut['mg_mag'], ls='', marker='o', c='grey', alpha=alpha, label="GC RGB")
ax_.plot(angc_['cg_mag'], angc_['mg_mag'], ls='', marker='.', c='red', alpha=alpha, label="Andrae x GC")
ax_.axhline(mg_cut, ls=':', c='grey')
ax_.axvline(cg_cut, ls=':', c='grey')
ax_.legend()
ax_.set_xlabel(r"$\mathrm{BP} - \mathrm{RP}$ (corr.) [mag]")
ax_.set_ylabel(r"$M_\mathrm{G}$ [mag]")
ax_.yaxis.set_inverted(True)

ax_ = ax[1,1]
ax_.set_title("All stars")
linewidth = 2.0
alpha = 0.75
ax_.hist(gccut['mg_mag'], bins=mg_edges,
         histtype='step', linewidth=linewidth, color='grey', alpha=alpha, label="GC RGB")
ax_.hist(gccut['mg_mag'], bins=mg_edges, weights=gccut['selfun'],
         histtype='step', linewidth=linewidth, linestyle='dashed', color='grey', alpha=alpha, label="GC RGB + SF")
ax_.hist(angc_['mg_mag'],  bins=mg_edges,
         histtype='step', linewidth=linewidth, color='red', alpha=alpha, label="Andrae x GC")
ax_.legend()
ax_.set_xlabel(r"$M_\mathrm{G}$ [mag]")

ax_ = ax[1,3]
ax_.set_title("All stars")
ax_.plot(angc_['teff_xgboost'], angc_['logg_xgboost'], ls='', marker='.', c='red', alpha=alpha, label="Andrae x GC")
ax_.legend()
ax_.set_xlabel(r"teff_xgboost")
ax_.xaxis.set_inverted(True)
ax_.set_ylabel(r"logg_xgboost")
ax_.yaxis.set_inverted(True)

plt.savefig(fig_path + '/magnitudes.png', dpi=600)
plt.show()
plt.close()

# %%



# %%
# Correct cuts

mg_new_cut = 1.5

gcstars['cut_new_rgb'] = (gcstars['cg_mag'] > cg_cut) & (gcstars['mg_mag'] < mg_new_cut)
mask = gcstars['cut_new_rgb'] & gcstars['cut_r']
mask &= gcstars['cut_andrae']
gccut = gcstars[mask].copy()
print("# total cut GC stars:", len(gccut))

# Cross-match the Andrae stars with _selected_ GC stars
angc = an.merge(gccut, on='source_id', how='inner')
print("angc records:", len(angc))
print("   memory: %.1f MiB" % (angc.memory_usage().sum()/(1024**2)))

# %%



# %%
# Count stars

# !!!!!
# This cut corresponds roughly to mg_mag = 1.5
logg_cut = 3.0

count_gccut = gccut.groupby('vaname', observed=True)\
    .agg(n_gc=('source_id', 'count'), feh=('feh', 'mean'), D=('D', 'mean'), selfun=('selfun', 'mean'), n_gc_w=('selfun', 'sum'))
count_gccut.reset_index(inplace=True)
#count_gcrgb

mask = np.full(len(angc), True)
mask &= angc['logg_xgboost'] < logg_cut
angc_ = angc[mask]
count_ag = angc_.groupby('vaname', observed=True)\
    .agg(n_ag=('source_id', 'count'), mh_xgboost=('mh_xgboost', 'mean'))
count_ag.reset_index(inplace=True)
#count_ag

# %%
# Merge all together

count_agXgccut = pd.merge(count_ag, count_gccut, on='vaname', how='inner')
count_agXgccut.fillna(0.0, inplace=True)

# Clean out wring fehs
mask = count_agXgccut['feh'] > -10
count_agXgccut = count_agXgccut[mask].copy()
count_agXgccut

# %%
# Plot statistics

count_agXgccut['rate']   = count_agXgccut['n_ag'] / count_agXgccut['n_gc']
count_agXgccut['rate_w'] = count_agXgccut['n_ag'] / (count_agXgccut['selfun'] * count_agXgccut['n_gc'])
count_agXgccut['rate_w_alt'] = count_agXgccut['n_ag'] / count_agXgccut['n_gc_w']
print(count_agXgccut[['rate', 'rate_w', 'selfun']].describe())


#_, ax = fig_init(shape=(2, 1), dims=(4.0, 4.5), font_size=5.0, layout='tight')
_, ax = fig_init(shape=(2, 1), dims=(4.0, 4.5), font_size=5.0)

#mpl.style.use('tableau-colorblind10')
plt.style.use('seaborn-v0_8-colorblind')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

alpha = 0.7
linewidth = 2.0

#rate_min = min(np.min(rate), np.min(rate/weight))
#rate_max = max(np.max(rate), np.max(rate/weight))
#rate_edges = np.linspace(rate_min, rate_max, 16)
rate_min, rate_max = 0.0, 2.0
rate_edges = np.linspace(rate_min, rate_max, 18)

ax_ = ax[0]
ax_.hist(count_agXgccut['rate_w'], bins=rate_edges, histtype='step',
         linewidth=linewidth, color=colors[0], alpha=alpha, label="Corrected")
ax_.hist(count_agXgccut['rate'],   bins=rate_edges, histtype='step',
         linewidth=linewidth, color=colors[2], alpha=alpha, label="Uncorrected")
#ax_.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), frameon=False)
ax_.legend(loc='upper center', frameon=False)
ax_.set_xlabel("rate")
ax_.set_ylim(ymax=50.0)

ax_ = ax[1]
#ax_.set_aspect(0.5)
sc = ax_.scatter(count_agXgccut['rate_w'], count_agXgccut['feh'], vmin=0.0,
                 c=count_agXgccut['D'], edgecolors=None, s=15.0, alpha=1.0)
cbar = plt.colorbar(sc, ax=ax_, location='top')
cbar.set_label("D [kpc]")
#
#vaname = 'NGC_6397'
#mask = count_agXgccut['vaname'] == vaname
#ax_.scatter(count_agXgccut[mask]['rate_w'], count_agXgccut[mask]['feh'], c='red', edgecolors=None, s=15.0)
#
ax_.set_xlabel("rate")
ax_.set_xlim(0.0, 2.0)
ax_.set_ylabel("[Fe/H]")

plt.savefig(fig_path + '/gc-andrae-rgb-rate-2.pdf')
plt.show()
plt.close()

# %%
