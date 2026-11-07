# %%

import sys
sys.path.append('..')
import re
import pickle

import importlib
from tqdm import tqdm

import numpy as np
import scipy as sp
from scipy.stats import qmc
import pandas as pd

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord

import matplotlib as mpl
import matplotlib.pyplot as plt

import utils
import config
from domain import Domain
import phys_models
import comp_models


show_figures = False

# %%



# %%
# Load everything

#
# Load subsample

print("\n* Load subsample")

df0 = pd.read_hdf(config.data_path + '/arc23aurgb.hdf')
print("Num. of subsample stars:", len(df0))

# Cut by [M/H]
mh_lo_lim = config.mh_lo_lim
mh_hi_lim = config.mh_hi_lim
mask = (mh_lo_lim <= df0['mh_xgboost']) & (df0['mh_xgboost'] < mh_hi_lim)

# Cut by log(g)
mask &= (df0['logg_xgboost'] < config.logg_cut)

df0 = df0[mask].copy()
print(f"Num. of subsample stars after selection:", len(df0))

# Downscale healpixel indices if necessary
if config.hpxorder < 7:
    df0['hpx'] = df0['hpx'] // 4**(7-config.hpxorder)

df = df0.copy()
del(df0)

#
# Load isochrones data

print("\n* Load isochrones data")

rgb_iso0 = pd.read_csv(f"../rgb_iso_logg_cut={config.logg_cut}.csv")

mask = (mh_lo_lim <= rgb_iso0['feh']) & (rgb_iso0['feh'] <= mh_hi_lim)
rgb_iso = rgb_iso0[mask].copy()

#
# Load domain

print("\n* Load domain")

dom = Domain().load(config.data_path + f"/domain_hpx{config.hpxorder}.npz")

#
# Load transfer function and auxiliaries

print("\n* Load transfer function and auxiliaries")

res = np.load(config.data_path + f"/transfun_hpx{config.hpxorder}.npz")
MG = res['MG']
SF = res['SF']
TF = res['TF']

print("shape(MG) =", np.shape(MG))
print("isnan(MG) =", np.any(np.isnan(MG)))
print("shape(SF) =", np.shape(SF))
print("isnan(SF) =", np.any(np.isnan(SF)))
print("shape(TF) =", np.shape(TF))
print("isnan(TF) =", np.any(np.isnan(TF)))

#SF_lbD = SF.mean(axis=-1)
#SF_lbD = np.median(SF, axis=-1)

#
# Load observables

print("\n* Load observables")

with open('observables.pkl', 'rb') as f:
    N_lbDG = pickle.load(f)
print("shape(N_lbDG) =", np.shape(N_lbDG))
print("isnan(N_lbDG) =", np.any(np.isnan(N_lbDG)))

N_lbD = N_lbDG.sum(axis=-1)
N_lbG = N_lbDG.sum(axis=-2)
print("sums:", N_lbDG.sum(), N_lbD.sum(), N_lbG.sum())

#
# Load estimates of Davies+2024 https://arxiv.org/abs/2410.21365

ell = pd.read_csv(config.data_path + '/elliot_radius_ins_acc_stats.csv')

#
# Load No-GSE model

#ph_nogse_r = pd.read_csv('../pl-nogse-0/ph_r.csv')

# %%



# %%
# Initialize the model

keys = config.keys

# GSE density
density_gse = config.density_gse

# Initialize loss function
xyz = dom.xyz
aux = {'TF': TF, 'MG': MG, 'dG': dom.dG, 'dVol': dom.dVol}
loss = phys_models.Loss_AuPL(xyz, density_gse, keys, aux)

# %%



# %%
# Apply the best model

print("\n* Apply the best model")

ast = pd.read_csv('model_de_ast.csv')
print("ast:")
print(ast)
theta = ast[keys].values[0]
params = dict(zip(keys, theta))
print("params =", params)

app = comp_models.apply_model_detailed(N_lbDG, loss, theta, density_gse)
Q_lbDG, Q_au_lbDG, Q_gse_lbDG = app['Q_lbDG'], app['Q_au_lbDG'], app['Q_gse_lbDG']
Q_lbD = Q_lbDG.sum(-1)
Q_lbG = Q_lbDG.sum(-2)

np.savez_compressed('observables_synth_de.npz', Q_au_lbDG=Q_au_lbDG, Q_gse_lbDG=Q_gse_lbDG)

# %%



# %%
# Plots

# Plot cuts and piercings
utils.plot_cuts(dom, N_lbDG, Q_lbDG, savefig_fname=f"fig/model-cuts-{config.save_suffix}.pdf")

# Plot sky projections and residuals
N_lb = N_lbDG.sum(axis=(-1, -2))
Q_lb = Q_lbDG.sum(axis=(-1, -2))
utils.plot_sky_projections(dom, N_lb, Q_lb, savefig_fname='fig/model_de_sky.pdf')

# Plot all projections and residuals
utils.plot_projections(dom, df, Q_lbD, xlim=(-20, 10), ylim=(-15, 15), zlim=(-15, 15),
                             savefig_fname=f"fig/model-proj-{config.save_suffix}.pdf")

# Plot xyz residuals
utils.plot_xyz_residuals(dom, N_lbD, Q_lbD, norm='symlog', savefig_fname=f"fig/model-proj-{config.save_suffix}-2.pdf")

# %%



# %%
# Normalize the density distributions as a PDF in the 'big' volume

print("\n* Normalize the density distributions as a PDF in the 'big' volume")

# More than enough
#r_min, r_max = 2.0, 70.0  # kpc
#r_min, r_max = dom.D_edges[0], dom.D_edges[-1]
r_min, r_max = dom.D_edges[0], 50.0
print("  r_min, r_max =", r_min, r_max)
Vol_big = 4*np.pi/3 * (r_max**3 - r_min**3)

sampler = qmc.Sobol(d=3, scramble=False)
xyzT = sampler.random_base2(m=25)
xyzT = qmc.scale(xyzT, [-r_max, -r_max, -r_max], [r_max, r_max, r_max])
print("  shape(xyzT)", xyzT.shape)

# Cut what's outside the spherical shell
mask = (r_min <= np.linalg.norm(xyzT, axis=1)) & (np.linalg.norm(xyzT, axis=1) <= r_max)
xyzT = xyzT[mask]
print("  shape(xyzT)", xyzT.shape)
n_qmc = xyzT.shape[0]
print("  n_qmc =", n_qmc)

rho_au_big_  = loss.density_au(xyzT[:,0], xyzT[:,1], xyzT[:,2], params)
rho_gse_big_ = density_gse(xyzT[:,0], xyzT[:,1], xyzT[:,2], params)
xi = params['xi']
rho_big_ = (1-xi)*rho_au_big_ + xi*rho_gse_big_

# Estimate the normalization with the MC integration
norm_big = 1 / (Vol_big/n_qmc * rho_big_.sum())
print("norm_big =", norm_big)
# Estimate mass fractions over the *big* volume
frac_au  = norm_big * Vol_big/n_qmc * (1-xi)*rho_au_big_ .sum()
frac_gse = norm_big * Vol_big/n_qmc *     xi*rho_gse_big_.sum()
print("frac_au = ", frac_au)
print("frac_gse =", frac_gse)
print("1 ~", norm_big * Vol_big/n_qmc * rho_big_.sum())

# %%
# Estimate the total number of stars

norm_dom = app['norm_dom']
print("norm_dom =", norm_dom)

# Total selection probability in the *domain* volume
Z = app['Z']
print(f"Z={Z:g}")

# Big-to-domain renormalization factor
C_dom = norm_dom / norm_big
print("norm_dom / norm_big =", norm_dom / norm_big)
assert (norm_dom / norm_big) > 1.0

N_sample = int(N_lbDG.sum())
# Number of RGB stars in the whole population inside the domain
N_rgb_dom = int(N_sample / Z)
# Number of RGB stars in the whole population inside the big volume
N_rgb = int(norm_dom / norm_big * N_rgb_dom)
print("N_sample =", N_sample)
print("N_rgb_dom =", N_rgb_dom)
print("N_rgb =", N_rgb)

# Fraction of RGB stars in the whole population, given the isochrone
n_rgb = rgb_iso['n_rgb'].mean()
print("<n_rgb> =", n_rgb)
# Number of stars in the whole population inside the big volume
N_pop = N_rgb / n_rgb
print("N_pop  =", N_pop)

# Number of Aurora stars in the whole population (inside the big volume)
N_pop_au  = int(frac_au * N_pop)
# Number of GSE stars in the whole population (inside the big volume)
N_pop_gse = int(frac_gse * N_pop)
print("N_pop_au  =", N_pop_au)
print("N_pop_gse =", N_pop_gse)

# %%
# Estimate masses

mmean_iso = rgb_iso['mmean_iso'].mean()
print("<mmean_iso> =", mmean_iso)
M_tot = mmean_iso * N_pop
print(f"M_tot = {M_tot:.2e} M_sun")
mmean_rgb = rgb_iso['mmean_rgb'].mean()
print("<mmean_rgb> =", mmean_rgb)
M_rgb = mmean_rgb * N_rgb
print(f"M_rgb = {M_rgb:.2e} M_sun")
M_tot_au  = frac_au * M_tot
M_tot_gse = frac_gse * M_tot
print(f"M_tot_au  = {M_tot_au:.2e} M_sun")
print(f"M_tot_gse = {M_tot_gse:.2e} M_sun")

# %%
# Save the stars number and mass estimates

out__ = {'name': f"{mh_lo_lim:.1f} < [M/H] < {mh_hi_lim:.1f}"}
out__.update(params)
out__.update({'N_sample': N_sample, 'N_rgb_dom': N_rgb_dom, 'N_rgb': N_rgb, 'N_pop': N_pop,
              'M_rgb': M_rgb, 'M_tot': M_tot, 'M_tot_au': M_tot_au, 'M_tot_gse': M_tot_gse})
out = pd.DataFrame([out__])
out.to_csv('out.csv', mode='w', index=False)

# %%



# %%
# Calculate theoretical density distributions in the 'big' volume

# PDFs in the *big* volume
rho_au_big  = norm_big * (1-xi)*rho_au_big_
rho_gse_big = norm_big *     xi*rho_gse_big_
rho_big = rho_au_big + rho_gse_big

# Keep it
ph = pd.DataFrame({'x': xyzT[:,0], 'y': xyzT[:,1], 'z': xyzT[:,2],
                   'rho_gse': rho_gse_big, 'rho_au': rho_au_big, 'rho': rho_big})

# %%
# Calculate additional profiles in the 'big' volume

importlib.reload(comp_models)

ph = comp_models.get_horta_etc(xyzT, ph)

# %%
# Evaluate spherically-symmetrized radial density profiles

ph['r'] = np.sqrt(ph['x']**2 + ph['y']**2 + ph['z']**2)
ph['r_bins'] = pd.cut(ph['r'], dom.D_edges, right=False, include_lowest=True)

ph_r = ph.groupby('r_bins', observed=True, as_index=False).mean()

# %%



# %%
# Evaluate the confidence level using the Wison Score Interval

def ci_wilson(X, z):
    N = X.sum()
    p = X / N
    var = z * np.sqrt(p*(1-p)*N + 0.25*z**2)
    ci_lo = (p*N + 0.5*z**2 - var) / (N + z**2)
    ci_hi = (p*N + 0.5*z**2 + var) / (N + z**2)
    return N*ci_lo, N*ci_hi

CI_lo_lbD, CI_hi_lbD = ci_wilson(Q_lbD, z=0.99)

# %%
# Bin radial profiles

dVol = np.full_like(dom.xyz[0], dom.dVol)
dom.r = np.sqrt(dom.xyz[0]**2 + dom.xyz[1]**2 + dom.xyz[2]**2)

Q_au_lbD  = Q_au_lbDG .sum(axis=-1)
Q_gse_lbD = Q_gse_lbDG.sum(axis=-1)

SF_lbD = np.mean(SF, axis=-1)
print("S_lbD min, max:", SF_lbD.min(), SF_lbD.max())

df = pd.DataFrame({'r': dom.r.flatten(), 'dVol': dVol.flatten(),
                   'N': N_lbD.flatten(), 'Q': Q_lbD.flatten(), 'Q_au': Q_au_lbD.flatten(), 'Q_gse': Q_gse_lbD.flatten(),
                   'SF': SF_lbD.flatten(), 'CI_lo': CI_lo_lbD.flatten(), 'CI_hi': CI_hi_lbD.flatten()})
df['r_bins'] = pd.cut(df['r'], dom.D_edges, right=False, include_lowest=True)

#df_r = df.groupby('r_bins', observed=True, as_index=False).sum()
df_r = df.groupby('r_bins', observed=True, as_index=False).mean()
rr = df.groupby('r_bins', observed=True, as_index=False)['r'].mean()
df_r['n'] = df_r['N'] / df_r['dVol']
df_r['q'] = df_r['Q'] / df_r['dVol']
df_r['q_au']  = df_r['Q_au']  / df_r['dVol']
df_r['q_gse'] = df_r['Q_gse'] / df_r['dVol']
df_r['ci_lo'] = df_r['CI_lo'] / df_r['dVol']
df_r['ci_hi'] = df_r['CI_hi'] / df_r['dVol']

r_l = df_r['r_bins'].map(lambda x: x.left)
r_r = df_r['r_bins'].map(lambda x: x.right)
r_edges = np.unique(np.array(list(r_l) + list(r_r)))

# %%
# Plot radial profiles

plt.rc('font', size=5.0)
plt.rc('lines', linewidth=1.2)
inch = 2.54  ## cm
width, height = 8/inch, 8/inch
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(width, height), layout='constrained', gridspec_kw={'height_ratios': [5, 1]})

mpl.style.use('tableau-colorblind10')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

alpha = 0.8

ax_ = ax0
ax_.set_title(f"{mh_lo_lim} < [M/H] < {mh_hi_lim}")
#
ax_.plot(dom.D, df_r['q'], 'k', alpha=alpha, label="Model & selection")
#ax_.fill_between(dom.D, df_r['ci_lo'], df_r['ci_hi'], fc='k', alpha=0.25)
#ax_.plot(dom.D, df_r['q'] / df_r['SF'], 'k', alpha=alpha, label="Model & selection / SF")
#
ax_.plot(dom.D, df_r['n'], c=colors[0], alpha=alpha, label="Data")
#ax_.plot(dom.D, df_r['n'] / df_r['SF'], c=colors[0], alpha=alpha, label="Data / SF")
#
#ax_.plot(ph_r['r'], N_rgb*ph_r['rho_au'], ls='--', c=colors[1], label="Model")
#ax_.plot(ph_r['r'], N_rgb*ph_r['rho_gse'], ls='--', c='tab:purple', label="GSE")
ax_.plot(ph_r['r'], N_rgb*ph_r['rho'], ls='--', c='k', label="Model")
#
#ax_.stairs(10**(4.5)*gcpop['numdens'], gcpop['r_edges'], color='g', alpha=alpha, label=r"GCs ($\times 10^{4.5}$)")
#
ax_.legend(handlelength=3.0, frameon=False)
#ax_.set_xlabel(r"Galactocentric distance [kpc]")
ax_.set_xlim(0.0, r_edges[-1])
ax_.set_ylabel(r"Number density of stars [kpc$^{-3}$]")
ax_.set_yscale('log')
ax_.set_ylim(1e-2, 1e4)

ax_ = ax1
ax_.axhline(0.0, c='k', lw=ax_.spines['top'].get_linewidth())
#ax_.plot(dom.D, (df_r['q'] - df_r['n']) / (df_r['q'] + df_r['n']), c='gray')
ax_.step(dom.D, (df_r['q'] - df_r['n']) / (df_r['q'] + df_r['n']), where='post', c='gray')
ax_.set_xlabel(r"Galactocentric distance [kpc]")
ax_.set_xlim(0.0, r_edges[-1])
ax_.set_ylabel("Rel. residual")
ax_.set_ylim(-0.5, 0.5)
ax_.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
ax__ = ax_.twinx()
ax__.set_ylim(-0.5, 0.5)
ax__.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
ax__.set_yticklabels([])

plt.savefig(f"fig/model-radial-{config.save_suffix}.pdf")
if show_figures:  plt.show()
plt.close()

# %%
# Plot the radial profiles with gradients

plt.rc('font', size=5.0)
plt.rc('lines', linewidth=1.2)
inch = 2.54  ## cm
width, height = 8/inch, 6.4/inch
plt.figure(figsize=(width, height), layout='constrained')

mpl.style.use('tableau-colorblind10')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig, ax0 = plt.subplots(1, 1, figsize=(width, height), layout='constrained')

alpha = 0.8

ax_ = ax0
ax_.set_title(f"{mh_lo_lim} < [M/H] < {mh_hi_lim}")
# Deason et al. (2011)
mask_Deason = ph_r['r'] >= 10.0
ax_.plot(ph_r[mask_Deason]['r'], 1e4*ph_r[mask_Deason]['rho_Deason'], c='c', lw=3.0, alpha=0.75*alpha,
         label="A-type stars (Deason et al. 2011)")
#
ax_.plot(ph_r['r'], M_rgb*ph_r['rho'], ls='-', c='k', alpha=alpha, label="Model")
#ax_.plot(ph_r['r'], M_rgb*ph_r['rho_au'], ls='-', c=colors[1], alpha=alpha, label="Aurora")
#
#ax_.plot(ph_nogse_r['r'], ph_nogse_r['m*rho'], ls=':', c='grey', alpha=alpha, label="No GSE")
# Horta et al. (2021)
ax_.plot(ph_r['r'], 1e-3*ph_r['m*rho_H1'], c=colors[4], alpha=alpha, label="Halo giants (Horta et al. 2021)")
ax_.plot(ph_r['r'], 1e-2*ph_r['m*rho_H2'], c='#4daf4a', alpha=alpha, label="N-rich giants (Horta et al. 2021)")
#
#ax_.stairs(gcpop['massdens'], gcpop['r_edges'], color='g', alpha=alpha, label=r"GCs")
#
ax_.legend(handlelength=3.0, frameon=False)
ax_.set_xlabel(r"Galactocentric distance [kpc]")
ax_.set_xscale('log')
ax_.set_xlim(1.0, 20.0)
ax_.set_ylabel(r"Mass density of stars [$M_\odot$ kpc$^{-3}$]")
ax_.set_ylim(1e-2, 1e5)
ax_.set_yscale('log')

#
# Add the inset

left, bottom, width, height = [0.24, 0.2, 0.3, 0.3]
ax__ = fig.add_axes([left, bottom, width, height])
ax__.set_title("Slopes", size=5.5)

# Deason et al. (2011)
mask_Deason = ph_r['r'] >= 10.0
ax__.plot(ph_r[mask_Deason]['r'], np.gradient(np.log(2.6e6*ph_r['rho_Deason']), np.log(ph_r['r']))[mask_Deason], c='c', lw=2.5, alpha=0.75*alpha)
#
ax__.plot(ph_r['r'], np.gradient(np.log(ph_r['rho']), np.log(ph_r['r'])), ls='-', c='k', alpha=alpha)
#ax__.plot(ph_r['r'], np.gradient(np.log(ph_r['rho_au']), np.log(ph_r['r'])), ls='-', c=colors[1], alpha=alpha)
#
#ax__.plot(ph_nogse_r['r'], np.gradient(np.log(ph_nogse_r['m*rho']), np.log(ph_nogse_r['r'])), ls=':', c='grey', alpha=alpha)
# Horta et al. (2021)
ax__.plot(ph_r['r'], np.gradient(np.log(ph_r['m*rho_H1']), np.log(ph_r['r'])), c=colors[4], alpha=alpha)
ax__.plot(ph_r['r'], np.gradient(np.log(ph_r['m*rho_H2']), np.log(ph_r['r'])), c='#4daf4a', alpha=alpha)
#ax__.axhline(-4.47, alpha=0.4)
#
ax__.set_xscale('log')
ax__.set_xlim(1.0, 20.0)
ax__.set_xticklabels([])
ax__.set_ylim(-6, 0)
#ax__.set_yscale('log')

plt.savefig(f"fig/model-radial-horta-{config.save_suffix}.pdf")
if show_figures:  plt.show()
plt.close()

# %%
# Plot the radial profiles in log-log scales

plt.rc('font', size=5.0)
plt.rc('lines', linewidth=1.2)
inch = 2.54  ## cm
width, height = 8/inch, 6/inch
fig, ax0 = plt.subplots(1, 1, figsize=(width, height), layout='constrained')

mpl.style.use('tableau-colorblind10')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

alpha = 0.8

ax_ = ax0
ax_.set_title(f"{mh_lo_lim} < [M/H] < {mh_hi_lim}")
#
ax_.plot(dom.D, df_r['q'], 'k', alpha=alpha, label="Model & selection")
#
ax_.plot(dom.D, df_r['n'], c=colors[0], alpha=alpha, label="Data")
#
#ax_.plot(ph_r['r'], N_rgb*ph_r['rho_au'], ls='--', c=colors[1], label="Aurora")
#ax_.plot(ph_r['r'], N_rgb*ph_r['rho_gse'], ls='--', c=colors[6], label="GS/E")
ax_.plot(ph_r['r'], N_rgb*ph_r['rho'], ls='--', c='k', label="Model")
#
ax_.legend(handlelength=3.0, frameon=False)
ax_.set_xlabel(r"Galactocentric distance [kpc]")
ax_.set_xscale('log')
ax_.set_xlim(1.0, r_edges[-1])
ax_.set_ylabel(r"Number density of stars [kpc$^{-3}$]")
ax_.set_yscale('log')

plt.savefig(f"fig/model-radial-loglog-{config.save_suffix}.pdf")
if show_figures:  plt.show()
plt.close()

# %%
# Save radial profiles

ph_r['dlogrho_dlogr'] = np.gradient(np.log(ph_r['rho']), np.log(ph_r['r']))
ph_r['dlogrho_au_dlogr'] = np.gradient(np.log(ph_r['rho_au']), np.log(ph_r['r']))
ph_r.to_csv('ph_r.csv', mode='w', index=False)

# %%



# %%
# Plot luminosity function

_, ax = utils.fig_init(dims=(8.0, 6.0))

ax.set_title("Luminosity function")
M = np.linspace(-3.0, 2.0, 100)
ax.plot(M, phys_models.lumfun_linear(M, (params['lf1'], params['lf2'], params['lf3'])))
ax.set_xlabel(r"$\mathrm{M}_\mathrm{G}$ [mag]")

plt.savefig(f"fig/model-lf-{config.save_suffix}.pdf")
if show_figures:  plt.show()
plt.close()

# %%



# %%
# Estimate Fisher matrix

def get_sigma(theta, loss, N_lbDG, delta):

    #
    # Evaluate gradient for PMF and logPMF

    grad_p    = np.empty(np.asarray(theta).shape + (np.prod(N_lbDG.shape),))
    grad_logp = np.empty_like(grad_p)
    for i in range(len(theta)):
        theta_hi = theta.copy()
        theta_hi[i] *= 1 + delta
        theta_lo = theta.copy()
        theta_lo[i] *= 1 - delta

        p_hi = loss.pmf_lbDG(theta_hi, -np.inf, np.inf, phys_models.lumfun_linear).ravel()
        p_lo = loss.pmf_lbDG(theta_lo, -np.inf, np.inf, phys_models.lumfun_linear).ravel()

        logp_hi = np.zeros_like(p_hi)
        mask = p_hi > 0.0
        logp_hi[mask] = np.log(p_hi[mask])

        logp_lo = np.zeros_like(logp_hi)
        mask = p_lo > 0.0
        logp_lo[mask] = np.log(p_lo[mask])

        grad_p[i]    = (p_hi - p_lo) / (2*delta)
        grad_logp[i] = (logp_hi - logp_lo) / (2*delta)

    #
    # Evaluate the Fisher matrix

    F = np.full(theta.shape*2, np.nan)
    for i in range(len(theta)):
        for j in range(i+1):
            F[i, j] = 0.5*N_sample * ( grad_p[i] @ grad_logp[j] + grad_p[j] @ grad_logp[i] )
            F[j, i] = F[i, j]
    inv_F = np.linalg.pinv(F)
    sigma = np.sqrt(np.diag(inv_F))

    return sigma


#deltas = np.array([1e-4, 3e-4, 1e-3, 3e-2, 1e-2, 3e-2, 1e-1])
deltas = np.array([1e-3, 1e-2, 1e-1])
sigmas = np.full(deltas.shape + theta.shape, np.nan)
for i, delta in enumerate(deltas):
    sigmas[i] = get_sigma(theta, loss, N_lbDG, delta)
    print(f"{i}: delta = {delta:g}\tsigma = {sigmas[i]}")

print("theta =", theta)
sigma_theta = np.nanmean(sigmas, axis=0)
print("sigma_theta =", sigma_theta)

# %%
# Estimate the log-evidence

#lnL = loss.log_prob(theta, -np.inf, np.inf, phys_models.lumfun_linear, N_lbDG)
lnL = ast['lnL'].values[0]
print("lnL =", lnL)
lnZ = lnL + 0.5*np.nansum(np.log(sigma_theta))
print("lnZ =", lnZ)

# %%



# %%
# Nice output

print("------------")

def format_exp_number(x):
    m, e = re.split(r"e\+?", "%.2e" % x)
    return r"$%.2f\times10^{%d}$" % (float(m), int(e))

print("$%.1f$ \\dots $%.1f$" % (mh_lo_lim, mh_hi_lim),
      #"& $%d$" % -ast['lnL'].values[0],
      #"& $%d$" % round(-ast['lnL'].values[0]),
      "& $%d$" % round(lnZ),
      "& $%.2f\\pm%.2f$" % (theta[0]*theta[1], theta[0]*sigma_theta[1] + theta[1]*sigma_theta[0]),
      "& $%.2f\\pm%.2f$" % (theta[0], sigma_theta[0]),
      "& $%.2f\\pm%.2f$" % (theta[1], sigma_theta[1]),
      "& $%.3f\\pm%.3f$" % (theta[2], sigma_theta[2]),
      "& $%.3f\\pm%.3f$" % (theta[-3], sigma_theta[-3]),
      "& $%.3f\\pm%.3f$" % (theta[-2], sigma_theta[-2]),
      "& $%.3f\\pm%.3f$" % (theta[-1], sigma_theta[-1]),
      "& %s" % round(out['N_sample'].values[0]),
      "& %s" % round(out['N_rgb'].values[0]),
      "& %s" % format_exp_number(out['M_tot'].values[0]))

# %%
