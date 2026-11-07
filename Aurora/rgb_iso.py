# %%

import sys
from tqdm import tqdm

import numpy as np
from numpy import pi, sqrt, exp, log, log10
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import minimint
import imf

import config

# %%
# Download and initialize DB
# https://github.com/segasai/minimint

data_path = config.data_path + '/minimint_data'
#minimint.download_and_prepare(filters=['UBVRIplus'], outp_prefix=data_path)
mi = minimint.Interpolator(['Gaia_G_EDR3', 'Gaia_BP_EDR3', 'Gaia_RP_EDR3'], data_prefix=data_path)

kroupa = imf.Kroupa()

# %%

def get_stats(mi, mf, star_age, feh, logg_cut):
    # Sample stellar mases
    masses = np.random.uniform(0.75, 0.9, 30000)

    # Generate isochrones
    iso_ = [ mi(mass, log10(star_age), feh)  for mass in tqdm(masses) ]
    iso = pd.DataFrame(iso_)
    del(iso_)

    # Mass interval for RGB stars
    mask = (iso['phase'] == 2) & (iso['logg'] <= logg_cut)
    mmin_rgb = iso[mask]['mass'].min()
    mmax_rgb = iso[mask]['mass'].max()
    print("mmin_rgb =", mmin_rgb)
    print("mmax_rgb =", mmax_rgb)

    # Minimum stellar mass
    mmin_iso = 0.1
    #mmin_iso = 0.5

    # Max mass of a star that is still alive to the moment 'star_age'
    mmax_iso = mi.getMaxMass(log10(star_age), feh)
    print("mmax_iso =", mmax_iso)

    # We assume a minimal mass to be `0.1`, hence the IMF must be normalised to the range `[0.1, mmax_iso]`
    norm = 1 / kroupa.integrate(mmin_iso, mmax_iso)[0]

    # Mean mass of stars in population of the still alive stars
    # Use: M_pop = mmean_iso * N_pop
    mmean_iso = norm * kroupa.m_integrate(mmin_iso, mmax_iso)[0]
    print("mmean_iso =", mmean_iso)

    # Number fraction of RGB stars among all stars
    # Use: N_RGB = n_rgb * N_pop
    n_rgb = norm * kroupa.integrate(mmin_rgb, mmax_rgb)[0]
    print("n_rgb =", n_rgb)

    # Mean mass of the RGB stars
    # Use: M_RGB = mmean_rgb * N_RGB
    mmean_rgb = norm * kroupa.m_integrate(mmin_rgb, mmax_rgb)[0] / n_rgb
    print("mmean_rgb =", mmean_rgb)

    # Mass of the whole isochrone per one RGB star
    # Use: M_pop = m_iso_per_rgb * N_RGB
    m_iso_per_rgb = mmean_iso / n_rgb
    print("m_iso_per_rgb =", m_iso_per_rgb)

    d = { 'star_age': star_age, 'feh': feh,
          'mmin_iso': mmin_iso, 'mmax_iso': mmax_iso, 'mmin_rgb': mmin_rgb, 'mmax_rgb': mmax_rgb,
          'mmean_iso': mmean_iso, 'mmean_rgb': mmean_rgb, 'n_rgb': n_rgb,
          'm_iso_per_rgb': m_iso_per_rgb }

    return d, iso

# %%

#logg_cut = 3.5
logg_cut = 3.0
#logg_cut = 2.4
#logg_cut = 2.0
print("logg_cut =", logg_cut)

#ages = [10e9, 11e9, 12e9, 13e9, 13.7e9]
ages = [13e9]

fehs = [-3.0, -2.0, -1.5, -1.3, -1.0]
out_ = []
isos = []
for age in ages:
    for feh in fehs:
        print("***")
        print("age = %e [yr]" % age)
        print("feh =", feh)

        d, iso_ = get_stats(mi, kroupa, age, feh, logg_cut)
        #print("%.3g M_sun" % (d['frac'] * 7919265))

        out_.append(d)
        isos.append(iso_)

out = pd.DataFrame(out_)
print(out)
out.to_csv(f"rgb_iso_logg_cut={logg_cut}.csv", index=False)
#sys.exit(0)

# %%
# Plot

plt.rc('font', size=6.0)
plt.rc('lines', linewidth=1.2)
inch = 2.54  ## cm
width, height = 18/inch, 10*len(ages)/inch

_, ax = plt.subplots(len(ages), 2, figsize=(width, height), squeeze=False, layout='constrained')
ccycle = [ p['color'] for p in plt.rcParams['axes.prop_cycle'] ]
alpha = 1.0

for k in range(len(ages)):
    ax_ = ax[k,0]
    ax_.set_title("age = %g [yr]" % ages[k])

    ax_.axhline(logg_cut, ls=':', c='k')

    for i in range(len(fehs)):
        feh = fehs[i]
        iso = isos[i]
        mask = iso['phase'] == 2
        ax_.plot(iso[mask]['mass'], iso[mask]['logg'], ls='', marker='.', ms=5.0, c=ccycle[i], alpha=alpha, label=f"[Fe/H]={feh:.1f}")
        ax_.plot(iso[~mask]['mass'], iso[~mask]['logg'], ls='', marker='.', ms=0.5, c=ccycle[i], alpha=alpha)
ax_.legend()
ax_.set_xlabel("Initial stellar mass [M_sun]")
ax_.set_ylabel("log g")

for k in range(len(ages)):
    ax_ = ax[k,1]
    ax_.set_title("age = %g [yr]" % ages[k])

    ax_.axvline(logg_cut, ls=':', c='k')

    for i in range(len(fehs)):
        feh = fehs[i]
        iso = isos[i]
        mask = iso['phase'] == 2
        ax_.hist(iso[mask]['logg'], histtype='step', linewidth=2.0, fc=ccycle[i], alpha=alpha, label=f"[Fe/H]={feh:.1f}")
        #ax_.hist(iso[~mask]['logg'], histtype='step', linewidth=1.0, fc=ccycle[i], alpha=alpha)
ax_.legend()
ax_.set_xlabel("log(g)")

plt.savefig('fig/rgb_iso.png', dpi=600)
plt.show()
plt.close()

# %%
