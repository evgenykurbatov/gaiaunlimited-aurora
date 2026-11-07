# %%

import sys
sys.path.append('..')
import time

from tqdm import tqdm

import numpy as np
import scipy as sp
from scipy.optimize import differential_evolution, minimize
import pandas as pd

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, Galactocentric

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import utils
import phys_models



def optimize_model_grad(N_lbDG, loss, low, high, theta_0, numiter=1, method=None):
    print("\noptimize_model_grad")

    t0 = time.time()

    theta = theta_0
    for _ in tqdm(range(numiter)):
        res = minimize(fun=loss.log_loss, x0=theta, tol=1e-6, bounds=list(zip(low, high)),
                       args=(low, high, phys_models.lumfun_linear, N_lbDG), method=method)
        log_like, theta = -res.fun, res.x

    ast__ = {'lnL': log_like, 'N': N_lbDG.sum()}
    ast__.update(dict(zip(loss.keys, theta)))
    ast = pd.DataFrame([ast__])

    print("Elapsed time:", time.strftime("%Hh%Mm%Ss", time.gmtime(time.time()-t0)))
    print("...done")
    
    return ast



def optimize_model_de(N_lbDG, loss, low, high, rndseed=42, numiter=1, **optparams):
    print("\noptimize_model_de")
    print("Parameters to optimize:", loss.keys)

    class IntermediateResult:
        """The callback function keeps the running time
        """

        t_ini  = None
        i = None

        def __init__(self):
            self.t_ini = time.time()
            self.i = 0

        def __call__(self, intermediate_result):
            print("step", self.i, "| t =", time.strftime("%Mm%Ss", time.gmtime(time.time()-self.t_ini)),
                  "| fun =", intermediate_result.fun)
            print("\tx =", intermediate_result.x)
            self.i += 1
            return False


    t0 = time.time()

    callback = IntermediateResult()
    #callback = None

    sims_ = []
    bounds = list(zip(low, high))
    for i in tqdm(range(numiter)):
        np.random.default_rng(rndseed + i)
        res = differential_evolution(
            loss.log_loss, bounds, args=(low, high, phys_models.lumfun_linear, N_lbDG),
            callback=callback, popsize=optparams['popsize'],
            tol=optparams['tol'], maxiter=optparams['maxiter'], updating='deferred', workers=optparams['workers'])
        log_like, theta = -res.fun, res.x

        sims__ = {'lnL': log_like, 'N': N_lbDG.sum()}
        sims__.update(dict(zip(loss.keys, theta)))
        sims_.append(sims__)
        sims = pd.DataFrame(sims_)

    ast = pd.DataFrame(sims.iloc[sims['lnL'].idxmax()]).T

    print("Elapsed time:", time.strftime("%Hh%Mm%Ss", time.gmtime(time.time()-t0)))
    print("...done")

    return ast



def optimize_model_nest(N_lbDG, loss, low, high, checkpointfile=None, rndseed=42, **optparams):
    print("\noptimize_model_nest")



def apply_model(N_lbDG, loss, theta):
    print("\napply_model")

    logloss = loss.log_loss(theta, -np.inf, np.inf, phys_models.lumfun_linear, N_lbDG)
    print("logloss =", logloss)

    # PMF in the observables' space, after selection applied
    p = loss.pmf_lbDG(theta, -np.inf, np.inf, phys_models.lumfun_linear)
    Q_lbDG = N_lbDG.sum() * p
    print("shape(q_lbDG) =", Q_lbDG.shape)
    print("sum(Q_lbDG) =", Q_lbDG.sum())

    return Q_lbDG



def apply_model_detailed(N_lbDG, loss, theta, density_gse=None):
    print("\napply_model_detailed")
    res = {}

    params = dict(zip(loss.keys, theta))

    logloss = loss.log_loss(theta, -np.inf, np.inf, phys_models.lumfun_linear, N_lbDG)
    print("logloss =", logloss)

    # Dims: hpx, D'
    xi = params['xi']
    rho_au_dom_  = loss.density_au(loss.xyz[0], loss.xyz[1], loss.xyz[2], params)
    if density_gse:
        rho_gse_dom_ = density_gse(loss.xyz[0], loss.xyz[1], loss.xyz[2], params)
    else:
        rho_gse_dom_ = loss.density_gse(loss.xyz[0], loss.xyz[1], loss.xyz[2], params)
    rho_dom_ = (1-xi)*rho_au_dom_ + xi*rho_gse_dom_

    # This will be PMF over the (hpx, D') space
    # Dims: hpx, D'
    norm_q = 1 / (rho_dom_ * loss.dVol).sum()
    q_au  = (1-xi) * norm_q * rho_au_dom_  * loss.dVol
    q_gse =     xi * norm_q * rho_gse_dom_ * loss.dVol
    q = norm_q * rho_dom_ * loss.dVol
    print("all close q:", np.allclose(q_au + q_gse, q))
    res['norm_dom'] = norm_q

    # Dims: hpx, G, D'
    Phi = phys_models.lumfun_linear(loss.MG, (params['lf1'], params['lf2'], params['lf3']))
    # This will be PMF over the G space
    dPsi_ = np.einsum('...kl,k->...kl', Phi, loss.dG)
    # Normalize over G
    # Dims: hpx, D'
    norm = dPsi_.sum(axis=-2)
    mask = norm > 0.0
    norm_inv = np.zeros_like(norm)
    norm_inv[mask] = 1./norm[mask]
    # Note: there might be no 'hpx' index
    # Dims: hpx, G, D'
    dPsi = np.einsum('...kl,...l->...kl', dPsi_, norm_inv)

    # Total selection probability in the *domain* volume
    Z = np.einsum('...jkl,...l,...kl->...jk', loss.TF, q, dPsi).sum()
    res['Z'] = Z

    # PMF in the observables' space, after selection applied
    # Dims: hpx, G
    # Note: there might be no 'hpx' index
    p_au_lbDG  = np.einsum('...jkl,...l,...kl->...jk', loss.TF, q_au,  dPsi)
    p_gse_lbDG = np.einsum('...jkl,...l,...kl->...jk', loss.TF, q_gse, dPsi)
    norm_p = 1 / (p_au_lbDG + p_gse_lbDG).sum()
    p_au_lbDG  *= norm_p
    p_gse_lbDG *= norm_p
    Q_au_lbDG  = N_lbDG.sum() * p_au_lbDG
    Q_gse_lbDG = N_lbDG.sum() * p_gse_lbDG

    # Dims: hpx, D, G
    # PMF in the observables' space, after selection applied
    p_lbDG = loss.pmf_lbDG(theta, -np.inf, np.inf, phys_models.lumfun_linear)
    Q_lbDG = N_lbDG.sum() * p_lbDG

    Q_lbD = Q_lbDG.sum(axis=-1)
    Q_lbG = Q_lbDG.sum(axis=-2)
    print(Q_lbDG.sum(), Q_lbD.sum(), Q_lbG.sum())
    print("all close Q:", np.allclose(Q_au_lbDG + Q_gse_lbDG, Q_lbDG))
    res['Q_au_lbDG']  = Q_au_lbDG
    res['Q_gse_lbDG'] = Q_gse_lbDG
    res['Q_lbDG']     = Q_lbDG

    return res



def get_horta_etc(xyzT, ph):
    print("\nget_horta_etc")

    #
    # Evaluate models of Horta et al. (2021, doi:10.1093/mnras/staa3598)

    # Halo giants | Triaxial SPL w/ disc+ cut-off

    def fn_r_H1(r):
        alpha = 3.48
        beta, fdisc = 0.01, 0.08
        return (1-fdisc) * r**(-alpha) * np.exp(-beta*r)

    M_H1 = 8.3e8  # M_sun
    _r_ = np.linspace(1.5, 15.0, 300)
    C_H1 = M_H1 / np.trapezoid(4*np.pi*_r_**2 * fn_r_H1(_r_), _r_)

    def fn_xyz_H1(x, y, z):
        alpha, p, q = 3.48, 0.76, 0.46
        beta, fdisc = 0.01, 0.08
        r = np.sqrt(x**2 + (y/p)**2 + (z/q)**2)
        return C_H1 * (1-fdisc) * r**(-alpha) * np.exp(-beta*r)

    # N-rich stars | Axisymmetric SPL

    def fn_r_H2(r):
        alpha = 4.47
        return r**(-alpha)

    M_H2 = 6.4e7  # M_sun
    _r_ = np.linspace(1.5, 15.0, 300)
    C_H2 = M_H2 / np.trapezoid(4*np.pi*_r_**2 * fn_r_H2(_r_), _r_)

    def fn_xyz_H2(x, y, z):
        alpha, q = 4.47, 0.47
        r = np.sqrt(x**2 + y**2 + (z/q)**2)
        return C_H2 * r**(-alpha)

    mrho_H1_ = fn_xyz_H1(xyzT[:,0], xyzT[:,1], xyzT[:,2])
    mrho_H2_ = fn_xyz_H2(xyzT[:,0], xyzT[:,1], xyzT[:,2])
    ph['m*rho_H1'] = mrho_H1_
    ph['m*rho_H2'] = mrho_H2_

    #
    # Evaluate a model of Deason et al. (2011, doi:10.1111/j.1365-2966.2011.19237.x)

    def fn_xyz_Deason(x, y, z):
        alpha, q = 2.3, 0.6
        m = np.sqrt(x**2 + y**2 + (z/q)**2)
        return m**(-alpha)

    rho_Deason_ = fn_xyz_Deason(xyzT[:,0], xyzT[:,1], xyzT[:,2])
    ph['rho_Deason'] = rho_Deason_

    return ph



def plot_hpx_profiles(dom, N_lbDG, Q_lbDG, hpx, params, savefig_fname=None, show=False):

    N_lbD = N_lbDG.sum(-1)
    N_lbG = N_lbDG.sum(-2)
    Q_lbD = Q_lbDG.sum(-1)
    Q_lbG = Q_lbDG.sum(-2)


    fig, ax = utils.fig_init(shape=(1, 3))
    fig.suptitle(f"Luminosity functions in a HEALPixel {hpx}")

    ax_ = ax[0]
    ax_.plot(dom.G, N_lbG[hpx], '.-', label="Data")
    ax_.plot(dom.G, Q_lbG[hpx], label="Model")
    ax_.legend(frameon=False)
    ax_.set_xlabel(r"$\mathrm{G}$ [mag]")

    ax_ = ax[1]
    ax_.plot(dom.D, N_lbD[hpx], '.-', label="Data")
    ax_.plot(dom.D, Q_lbD[hpx], label="Model")
    ax_.legend(frameon=False)
    ax_.set_xlabel(r"$D$ [kpc]")

    ax_ = ax[2]
    M = np.linspace(-4.0, 3.0, 71)
    ax_.plot(M, phys_models.lumfun_linear(M, (params['lf1'], params['lf2'], params['lf3'])), label="Model")
    ax_.legend(frameon=False)
    ax_.set_xlabel(r"$\mathrm{M}_\mathrm{G}$ [mag]")

    if savefig_fname:
        plt.savefig(savefig_fname)
    if show:  plt.show()
    plt.close()



def plot_corner_sims(keys, sims, savefig_fname=None, show=False):

    fig, ax = utils.fig_init(shape=(len(keys), len(keys)), dims=(3., 3.), fontsize=5.0)

    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if j < i:
                ax[i,j].axis('off')
                continue

            ax_ = ax[i,j]
            if i == j:
                ax_.hist(sims[ki], bins=5)
                ax_.set_xlabel(ki)
            else:
                ax_.plot(sims[ki], sims[kj], '.', alpha=0.25)
                ax_.set_xlabel(ki)
                ax_.set_ylabel(kj)

    if savefig_fname:
        plt.savefig(savefig_fname)
    if show:  plt.show()
    plt.close()
