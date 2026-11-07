# %%

import re
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import phys_models

# %%
#

def plot_summary(prefix, suffix, keys, nogse=False, title=None):

    print(f"Proscessing '{prefix}' ...")

    #
    # Load models

    mods_ = []
    mods_.append(pd.read_csv(f"{prefix}-1{suffix}/out.csv"))
    mods_.append(pd.read_csv(f"{prefix}-2{suffix}/out.csv"))
    mods_.append(pd.read_csv(f"{prefix}-3{suffix}/out.csv"))
    mods_.append(pd.read_csv(f"{prefix}-4{suffix}/out.csv"))

    mods = pd.concat(mods_)
    mods[keys['a']] = mods[keys['b']]*mods[keys['s']]

    print(mods)
    print("\nMedians:")
    if nogse:
        print(mods[['M_tot']].describe())
    else:
        print(mods[['M_tot', 'M_tot_au', 'M_tot_gse']].describe())

    modw = pd.read_csv(f"{prefix}-0{suffix}/out.csv")
    modw[keys['a']] = modw[keys['b']]*modw[keys['s']]

    #
    #

    def plot_hist(ax_, mods, column, color=None, label=None, xlabel=False, ls=None):
        for i, (_, mod) in enumerate(mods.iterrows()):
            edges = [ float(e)  for e in re.findall(r"[-+]?(?:\d*\.*\d+)", mod['name']) ]

            eb = ax_.errorbar(x=[0.5*(edges[0]+edges[1])], y=[mod[column]],
                              xerr=[0.5*(edges[1]-edges[0])], capsize=1.5, ecolor=color)
            if ls:
                eb[-1][0].set_linestyle(ls)

        ax_.set_xticks([-3.0, -2.0, -1.5, -1.3, -1.0])
        ax_.tick_params(axis='x', labelrotation=45)


    plt.rc('font', size=5.0)
    #plt.rc('axes', linewidth=0.6)
    #plt.rc('lines', linewidth=0.75)
    inch = 2.54  ## cm
    width, height = 8/inch, 11/inch
    #_, ax = plt.subplots(2, 2, figsize=(width, height), layout='constrained')
    #fig = plt.figure(figsize=(width, height), layout='constrained')
    fig = plt.figure(figsize=(width, height), layout='tight')
    fig.suptitle(title)

    mpl.style.use('tableau-colorblind10')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    #ls_w = None
    ls_w = ':'

    ax_ = fig.add_subplot(3, 2, 1, box_aspect=1.0)
    #
    plot_hist(ax_, mods, keys['a'], color=colors[4], label=r"$a_1$")
    plot_hist(ax_, mods, keys['b'], color=colors[5], label=r"$b_1$")
    plot_hist(ax_, modw, keys['a'], color=colors[4], label=r"$a_1$", ls=ls_w)
    plot_hist(ax_, modw, keys['b'], color=colors[5], label=r"$b_1$", ls=ls_w)
    #
    ax_.set_xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
    ax_.set_ylabel("$a_1$, $b_1$ [kpc]")
    #ax_.set_ylim(ymin=0.0)
    ax_.set_yscale('log')
    ax_.set_aspect(1.0/ax_.get_data_ratio(), adjustable='box')

    ax_ = fig.add_subplot(3, 2, 2, box_aspect=1.0)
    #
    plot_hist(ax_, mods, keys['s'], color='g')
    plot_hist(ax_, modw, keys['s'], color='g', ls=ls_w)
    #
    ax_.set_xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
    ax_.set_ylabel(r"$a_1/b_1$")
    ax_.set_ylim(1.0, 3.0)

    ax_ = fig.add_subplot(3, 2, 3, box_aspect=1.0)
    #
    plot_hist(ax_, mods, keys['k'], color='tab:green')
    plot_hist(ax_, modw, keys['k'], color='tab:green', ls=ls_w)
    if nogse:
        ax_.set_ylim(ymax=4.6)
    #
    ax_.set_xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
    ax_.set_ylabel(r"$k_1$")
    #ax_.set_ylim(3.0, 4.5)

    ax_ = fig.add_subplot(3, 2, 4, box_aspect=1.0)
    #
    if nogse:
        plot_hist(ax_, mods, 'M_tot', color='k')
        plot_hist(ax_, modw, 'M_tot', color='k', ls=ls_w)
        ax_.set_ylabel(r"$M_\mathrm{tot}$ [$M_\odot$]")
        ax_.set_ylim(1e7, 1e9)
    else:
        plot_hist(ax_, mods, 'M_tot_au', color=colors[1])
        plot_hist(ax_, modw, 'M_tot_au', color=colors[1], ls=ls_w)
        plot_hist(ax_, mods, 'M_tot_gse', color='tab:purple')
        plot_hist(ax_, modw, 'M_tot_gse', color='tab:purple', ls=ls_w)
        ax_.set_ylabel(r"$M_\mathrm{Au}$, $M_\mathrm{GSE}$ [$M_\odot$]")
        ax_.set_ylim(3e6, 1e9)
    #
    ax_.set_xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
    ax_.set_yscale('log')

    ax_ = fig.add_subplot(3, 1, 3, box_aspect=0.4)
    #
    MG = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    #
    names = mods['name'].values
    lf1, lf2, lf3 = mods['lf1'].values, mods['lf2'].values, mods['lf3'].values
    for i in range(len(lf1)):
        ax_.plot(MG, phys_models.lumfun_linear(MG, [lf1[i], lf2[i], lf3[i]]), label=f"{names[i]}")
    #
    names = modw['name'].values
    lf1, lf2, lf3 = modw['lf1'].values, modw['lf2'].values, modw['lf3'].values
    ax_.plot(MG, phys_models.lumfun_linear(MG, [lf1[0], lf2[0], lf3[0]]), ':k', label=f"{names[0]}")
    #
    ax_.legend(frameon=False)
    ax_.set_xlabel(r"$\mathrm{M}_\mathrm{G}$ [mag]")
    ax_.set_ylabel("Luminosity function")
    #ax_.set_ylim(ymax=0.42)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)

    plt.savefig(f"fig/model-summary-{prefix}{suffix}.pdf")
    #plt.show()
    plt.close()



keys = { 'a': 'a', 'b': 'b', 's': 's', 'k': 'k' }
plot_summary(prefix='pl-sc-hpx4', suffix='', keys=keys, title="Aurora & `SC' GSE")
plot_summary(prefix='pl-scd-hpx4', suffix='', keys=keys, title="Aurora & `SC+D' GSE")
plot_summary(prefix='pl-nogse-hpx4', suffix='', keys=keys, nogse=True, title="One component")

keys = { 'a': 'a1', 'b': 'b1', 's': 's1', 'k': 'k1' }
plot_summary(prefix='pl-freesc-hpx4', suffix='', keys=keys, title="Aurora & `Free' GSE")
