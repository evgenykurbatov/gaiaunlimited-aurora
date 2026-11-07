# %%

import numpy as np

from domain import Domain

import config

# %%
# Setup domain
#

print("\n* Setup domain")

dom = Domain().make(config.nside, D_lim=(0.1, 18.0), D_num=42)
dom._info = f"HEALPix {config.hpxorder}; whole sky"

fpath = config.data_path + f"/domain_hpx{config.hpxorder}.npz"
dom.save(fpath)

print()
print("  hpxorder =", config.hpxorder)
print(f"  Omega = {dom.Omega:.2e} [sr] = ({np.sqrt(dom.Omega):.2e} [rad])^2 = ({np.sqrt(dom.Omega)*180/np.pi:.2f} [deg])^2")

# %%
