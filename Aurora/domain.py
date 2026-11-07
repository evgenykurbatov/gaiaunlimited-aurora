# %%

from tqdm import tqdm

import numpy as np
import pandas as pd

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, ICRS, Galactocentric

import matplotlib as mpl
import matplotlib.pyplot as plt

import global_config as config


class Domain():

    def __init__(self):
        pass


    def make(self, nside, D_lim, D_num, radius=None, verbose=True):

        #
        # Sky

        self.nside = nside

        hp = HEALPix(nside=nside, order='nested', frame=ICRS())
        self.hpxorder = int(np.log2(nside))
        self.npix = 12 * 4**self.hpxorder

        # Pixel indices
        if radius is None:
            self.ipix = np.arange(self.npix)
        else:
            co = SkyCoord(0.0, 0.0, frame='galactic', unit='deg')
            self.ipix = hp.cone_search_skycoord(co, radius=radius)

        # Pixel centres
        co = hp.healpix_to_skycoord(self.ipix)
        self.l = co.galactic.l.value
        self.b = co.galactic.b.value
        # Pixel areas
        self.Omega = hp.pixel_area.value

        #
        # Distances

        # Edges of the distance grid
        self.D_edges = np.linspace(D_lim[0], D_lim[1], D_num)
        # Centers of the distance grid
        self.D = 0.5*(self.D_edges[:-1] + self.D_edges[1:])
        # Steps of the distance grid
        self.dD = self.D_edges[1:] - self.D_edges[:-1]

        #
        # Magnitudes

        # Edges of the magnitude grid
        self.G_edges = config.G_edges
        # Centres of the magnitude grid
        self.G = config.G
        # Steps
        self.dG = self.G_edges[1:] - self.G_edges[:-1]

        #
        # Measures

        self.dVol = self.Omega * (self.D_edges[1:]**3 - self.D_edges[:-1]**3) / 3.0

        #
        # Projections

        ipix = self.ipix
        l = self.l * u.deg
        b = self.b * u.deg

        D = self.D

        xyz = np.empty((3, len(ipix), len(D)))

        if verbose:
            iterator = tqdm(zip(l, b), total=len(ipix))
        else:
            iterator = zip(l, b)

        for i, lb_ in enumerate(iterator):

            # Define coordinates in Galactic frame
            l_, b_ = lb_
            co_gal  = SkyCoord(l_, b_, distance=D*u.kpc, frame='galactic')

            # Transform to Galactocentric frame
            gc_frame = Galactocentric(galcen_distance=config.R_sun, z_sun=config.z_sun, galcen_v_sun=config.vel_sun)
            co_galc = co_gal.transform_to(gc_frame)

            # Rays
            xyz[0,i,:] = co_galc.x.value
            xyz[1,i,:] = co_galc.y.value
            xyz[2,i,:] = co_galc.z.value

        self.xyz = xyz

        return self


    def load(self, fpath):
        d = np.load(fpath)
        self.__dict__.update(d)

        return self


    def save(self, fpath):
        np.savez_compressed(fpath, **self.__dict__)
