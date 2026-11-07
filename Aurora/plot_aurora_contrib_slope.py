# %%

import numpy as np
from numpy import log10
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import config

# %%
# Configure plots

fig_path = 'fig'
fig_dpi = 600

#plt.style.use('tableau-colorblind10')
plt.style.use('seaborn-v0_8-colorblind')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def fig_init(dims=(8.0, 6.0), shape=(1, 1), font_size=7.0, layout='constrained', **kwargs):
    plt.rc('font', size=font_size)
    cm_per_in = 2.54
    fig, ax = plt.subplots(shape[1], shape[0],
                           figsize=(shape[0]*dims[0]/cm_per_in, shape[1]*dims[1]/cm_per_in),
                           layout=layout, **kwargs)
    return fig, ax

# %%

def process(prefix, title):

    print(f"Proscessing '{prefix}' ...")

    #
    # Load models

    mods = []
    mods.append(pd.read_csv(f"{prefix}-0/ph_r.csv"))
    mods.append(pd.read_csv(f"{prefix}-1/ph_r.csv"))
    mods.append(pd.read_csv(f"{prefix}-2/ph_r.csv"))
    mods.append(pd.read_csv(f"{prefix}-3/ph_r.csv"))
    mods.append(pd.read_csv(f"{prefix}-4/ph_r.csv"))

    #
    # Plot

    labels = [r"$-3.0 < [\text{M/H}] < -1.3$",
              r"$-3.0 < [\text{M/H}] < -2.0$",
              r"$-2.0 < [\text{M/H}] < -1.5$",
              r"$-1.5 < [\text{M/H}] < -1.3$",
              r"$-1.3 < [\text{M/H}] < -1.0$"]

    _, ax = fig_init(font_size=5.0)

    ax.set_title(title)
    ax.plot(elliot['r'], np.gradient(log10(elliot['mean_ins']), log10(elliot['r'])), marker='o', ls='', c='#555555', label="Davies+2024")
    for i in [1, 2, 3, 4]:
        ph_r = mods[i]
        ax.plot(ph_r['r'], np.gradient(log10(ph_r['rho_au'] / ph_r['rho']), log10(ph_r['r'])), label=labels[i])
    ax.plot(mods[0]['r'], np.gradient(log10(mods[0]['rho_au'] / mods[0]['rho']), log10(mods[0]['r'])), ':k', label=labels[0])
    ax.legend(loc='lower left', frameon=False)
    ax.set_xlabel("Galactocentric distance [kpc]")
    ax.set_xscale('log')
    ax.set_xlim(1.0, 18.0)
    ax.set_ylabel("Log-slope")
    #ax.set_ylim(0.0, 1.0)

    plt.savefig(fig_path + f"/model-aurora-contrib-slope-{prefix}.pdf")
    #plt.show()
    plt.close()

# %%
# Load data

elliot = pd.read_csv(config.data_path + '/elliot_radius_ins_acc_stats.csv')
print(elliot)

process('pl-sc', "Aurora + `SC' GSE")
process('pl-bpl', "Aurora + `BPL' GSE")
process('pl-dbpld', "Aurora + `DBPL+D' GSE")

# %%
