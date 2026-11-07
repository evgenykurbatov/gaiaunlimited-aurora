# %%

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import global_config as config
import utils

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

    #plt.style.use('tableau-colorblind10')
    plt.style.use('seaborn-v0_8-colorblind')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    _, ax = utils.fig_init(fontsize=5.0)

    ax.set_title(title)
    ax.errorbar(elliot['r'], elliot['mean_ins'], yerr=elliot['std_ins'], marker='o', ls='', c='#555555', label="Davies+2024")
    for i in [1, 2, 3, 4]:
        ph_r = mods[i]
        ax.plot(ph_r['r'], ph_r['rho_au'] / ph_r['rho'], label=labels[i])
    ax.plot(mods[0]['r'], mods[0]['rho_au'] / mods[0]['rho'], ':k', label=labels[0])
    #ax.legend(loc='lower left', frameon=False)
    ax.legend(ncol=2, loc='upper center', columnspacing=5.0, frameon=False)
    ax.set_xlabel("Galactocentric distance [kpc]")
    ax.set_xlim(0.0, 18.0)
    ax.set_ylabel("Relative amount of in-situ stars")
    ax.set_ylim(0.0, 1.1)

    plt.savefig(config.fig_path + f"/model-aurora-contrib-{prefix}.pdf")
    #plt.show()
    plt.close()

# %%
# Load data

elliot = pd.read_csv(config.data_path + '/elliot_radius_ins_acc_stats.csv')
print(elliot)

#process('pl-sc-hpx4', "Aurora & `SC' GSE")
#process('pl-scd-hpx4', "Aurora & `SC+D' GSE")
process('pl-freesc-hpx4', "Aurora & `Free' GSE")

# %%
