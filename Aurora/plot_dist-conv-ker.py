# %%

import importlib
from tqdm import tqdm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import global_config as config
import utils
importlib.reload(utils)

# %%

plt.rc('font', size=6.0)
inch = 2.54  ## cm
width, height = 8/inch, 6/inch
plt.figure(figsize=(width, height), layout='constrained')
#plt.suptitle(r"Parallax smoothing kernel at $D' = 8$ kpc")

D_ = 8.0
D = np.linspace(0.1, 16.0, 200)
K = lambda sigma_plx : np.exp(-0.5*(1/D-1/D_)**2/sigma_plx**2) / (np.sqrt(2*np.pi)*sigma_plx)

plt.axvline(D_, ls=':', c='k')
plt.plot(D, K(1e-2)/D**2, label=r"$\sigma_\varpi = 0.01$ mas")
plt.plot(D, K(2e-2)/D**2, label=r"$\sigma_\varpi = 0.02$ mas")
plt.plot(D, K(3e-2)/D**2, label=r"$\sigma_\varpi = 0.03$ mas")
plt.xlabel("$D$ [kpc]")
plt.legend(frameon=False)
plt.xlim(0.0, D[-1])
plt.ylim(ymin=0.0)

#plt.show()
plt.savefig('fig/model-dist-conv-ker.pdf')
plt.close()
