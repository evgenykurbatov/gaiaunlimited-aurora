# %%

import sys
sys.path.append('..')
import time
import pickle

import importlib
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import utils
import config
import phys_models
import comp_models
from domain import Domain

# %%
# Load everything

#
# Load domain

print("\n* Load domain")

dom = Domain().load(config.data_path + f"/domain_hpx{config.hpxorder}.npz")

#
# Load transfer function and auxiliaries

print("\n* Load transfer function and auxiliaries")

res = np.load(config.data_path + f"/transfun_hpx{config.hpxorder}.npz")
TF = res['TF']
MG = res['MG']

print("shape(TF) =", np.shape(TF))
print("isnan(TF) =", np.any(np.isnan(TF)))
print("shape(MG) =", np.shape(MG))
print("isnan(MG) =", np.any(np.isnan(MG)))

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

# %%



# %%
# Initialize the model

importlib.reload(phys_models)

keys = config.keys

# Initialize loss function
aux = {'TF': TF, 'MG': MG, 'dG': dom.dG, 'dVol': dom.dVol}
loss = phys_models.Loss_AuPL_FreeSCGSE(dom.xyz, keys, aux)

# Parameter bounds
b1_min, b1_max = 0.1, 5.0
s1_min, s1_max = 1.0, 10.0
k1_min, k1_max = 2.0, 6.0
a2_min, a2_max = 0.1, 50.0
b2_min, b2_max = 0.1, 50.0
c2_min, c2_max = 0.1, 50.0
k2_min, k2_max = 0.0, 5.0
r2_min, r2_max = 3.0, 50.0
theta_min, theta_max = 0.0, 360.0
eta_min, eta_max     = 0.0, 1.0
phi_min, phi_max     = 0.0, 180.0
xi_min, xi_max   = 0.0, 0.9
lf_min, lf_max   = 0.0, 0.99

low  = np.array([b1_min, s1_min, k1_min, a2_min, b2_min, c2_min, k2_min, r2_min, theta_min, eta_min, phi_min, xi_min, lf_min, lf_min, lf_min])
high = np.array([b1_max, s1_max, k1_max, a2_max, b2_max, c2_max, k2_max, r2_max, theta_max, eta_max, phi_max, xi_max, lf_max, lf_max, lf_max])

# Load previous solutions, if any
sims_ = []
simsfile = 'model_de.csv'
if simsfile:
    try:
        sims = pd.read_csv(simsfile)
        sims_ = sims.to_dict('records')
    except FileNotFoundError:
        pass
print("Previous steps:", len(sims_))

# %%
# Optimize model

optparams = {
    'tol': 1e-6,
    'popsize': 30,
    'maxiter': 100_000,
    'workers': 20,
}

try:

    # Use Differential Evolution to outline solutions
    ast = comp_models.optimize_model_de(N_lbDG, loss, low, high, rndseed=42, numiter=1, **optparams)
    
    # Use Gradient Descent to polish up the best solution
    theta = ast[keys].values[0]
    ast = comp_models.optimize_model_grad(N_lbDG, loss, low, high, theta, numiter=100)

    print("ast:", ast.to_dict('records'))

except Exception as e:
    print("ERROR:")
    print(e)
    sys.exit(-1)

# %%

# Save current run
sims_ += ast.to_dict('records')
sims = pd.DataFrame(sims_)
sims.to_csv(simsfile, mode='w', index=False)

# Find the best solutions over previous runs
ast = pd.DataFrame(sims.iloc[sims['lnL'].idxmax()]).T
ast.to_csv('model_de_ast.csv', mode='w', index=False)

# %%
# Apply the best model

print()

ast = pd.read_csv('model_de_ast.csv')
print("ast:")
print(ast)
theta = ast[keys].values[0]
params = dict(zip(keys, theta))
print("params =", params)

Q_lbDG = comp_models.apply_model(N_lbDG, loss, theta)

# %%
# Make some plots

importlib.reload(comp_models)

N_lb = N_lbDG.sum(axis=(-1, -2))
Q_lb = Q_lbDG.sum(axis=(-1, -2))
comp_models.plot_sky_projections(dom, N_lb, Q_lb, savefig_fname='fig/model_de_sky.pdf')

# bs = [-30, -12, -5, 5, 20]
# ls = [-60, -10, 0, 10, 20]
# hpx = [512 746 448 457 484]
hpx = 457
comp_models.plot_hpx_profiles(dom, N_lbDG, Q_lbDG, hpx, params, savefig_fname='fig/model_de_profiles.pdf')

if len(sims) > 1:
    comp_models.plot_corner_sims(keys, sims, savefig_fname='fig/model_de_corner_sims.pdf')

# %%
