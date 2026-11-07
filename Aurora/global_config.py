
import pathlib

import numpy as np

import astropy.units as u


#
#

fig_path = 'fig/'
fig_dpi = 600

data_root_path = str(pathlib.Path.home() / 'where-all-data-lives')
data_path = data_root_path + '/Data'

#
# WSDB

wsdb_host = 'wsdb_host'
wsdb_user = 'wsdb_user'
wsdb_password = 'wsdb_password'

#
# HEALPix

#hpxorder = 3
hpxorder = 4
#hpxorder = 5
hpxbase = 2**35 * 4**(12 - hpxorder)
nside = 2**hpxorder
npix = 12 * 4**hpxorder
print(f"hpxordfer={hpxorder} -> (nside={nside}, npix={npix})")

#
#

logg_cut = 3.0

#
# Magnitude bins

G_min, G_max, dG  = 2.0, 20.0, 0.5  ## [mag]
G_edges = np.linspace(G_min, G_max, int((G_max-G_min)/dG)+1)
G = 0.5*(G_edges[:-1] + G_edges[1:])

#
# Galactocentric frame

# From paper Ardern-Arentsen A et al. arXiv:2312.03847 (p6,7)
# Bland-Hawthorn & Gerhard 2016; Portail et al. 2017
R_sun = 8.2 * u.kpc

y_sun = 0.0 * u.kpc

# Bennett & Bovy (2019) https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1417B
z_sun = 20.8 * u.pc

pos_sun = u.Quantity([R_sun, y_sun, z_sun])

# DO NOT use it!
#vel_sun = [12.9, 245.6, 7.78] * u.km/u.s

# Schönrich et al. (2010); Reid & Brunthaler (2004)
vel_sun = [11.10, 246.6, 7.24] * u.km/u.s
