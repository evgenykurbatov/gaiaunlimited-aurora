
import numpy as np
import scipy as sp
import pandas as pd

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, ICRS, Galactic, Galactocentric

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm



def fig_init(dims=(8.0, 7.0), shape=(1, 1), fontsize=7.0, squeeze=True, **kwargs):
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    plt.rc('font', size=fontsize)
    cm_per_in = 2.54
    fig, ax = plt.subplots(shape[0], shape[1],  # nrows, ncols
                           figsize=(shape[1]*dims[0]/cm_per_in, shape[0]*dims[1]/cm_per_in),
                           layout='constrained', squeeze=squeeze, **kwargs)
    return fig, ax



def plot_hpxmap(sky, step=0.5, interp=True, norm='log', vmin=None, vmax=None,
                xlabel=True, ylabel=True, title=None, cmap='inferno', cbar=True, ax=None):

    npix = sky.shape[0]
    nside = int(np.sqrt(npix // 12))

    l_grid = np.arange(-180.0, 180.0, step)
    b_grid = np.arange(-90.0, 90.0, step)
    b_mesh, l_mesh = np.meshgrid(b_grid, l_grid)

    hp = HEALPix(nside=nside, order='nested', frame=ICRS())
    co = SkyCoord(l_mesh, b_mesh, frame='galactic', unit='deg')

    if interp:
        prb_mesh = hp.interpolate_bilinear_skycoord(co, sky)
    else:
        ipix = hp.skycoord_to_healpix(co)
        prb_mesh = sky[ipix]

    if ax is None:
        ax = plt.gca()

    """
    if vmin is None:
        vmin = np.nanmin(sky)
    if vmax is None:
        vmax = np.nanmax(sky)
    """

    if norm == 'log':
        if vmin is None:
            vmin = np.nanmin(sky)
        if vmax is None:
            vmax = np.nanmax(sky)
        if vmin == 0:
            vmin = 1e-6*vmax
        norm = LogNorm(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(l_grid, b_grid, prb_mesh.T,
                           cmap=cmap, norm=norm, rasterized=True)
    else:
        im = ax.pcolormesh(l_grid, b_grid, prb_mesh.T,
                           cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, rasterized=True)

    ax.set_xlim(180.0, -180.0)
    ax.set_ylim(-90, 90)
    if cbar:
        # Set Matplotlib colorbar size to match graph
        # https://stackoverflow.com/a/18195921/1038377
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        #
        """
        #https://stackoverflow.com/a/56900830/1038377
        # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
        # You can change 0.01 to adjust the distance between the main image and the colorbar.
        # You can change 0.02 to adjust the width of the colorbar.
        # This practice is universal for both subplots and GeoAxes.
        fig = ax.get_figure()
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        #
        """
        plt.colorbar(im, cax=cax)

    ax.set_title(title)
    if xlabel is True:
        ax.set_xlabel(r"$l$ [deg]")
    else:
        ax.set_xlabel(xlabel)
    if ylabel is True:
        ax.set_ylabel(r"$b$ [deg]")
    else:
        ax.set_ylabel(ylabel)

    return im



def plot_projections_old(df, xlim=None, ylim=None, zlim=None, bins=200, cmap='inferno'):

    x_sun = Galactocentric().galcen_distance.to('kpc').value
    y_sun = 0.0
    z_sun = Galactocentric().z_sun.to('kpc').value

    plt.rc('font', size=6.0)
    inch = 2.54  ## cm
    width, height = 17/inch, 4/inch
    plt.figure(figsize=(width, height), layout='constrained')

    ax = plt.subplot(1, 3, 1)
    l = (df['l'] + 180) % 360 - 180
    b = df['b']
    df_hist_lb, l_bins, b_bins = np.histogram2d(l, b, bins=bins)
    im = ax.pcolormesh(l_bins, b_bins, df_hist_lb.T, cmap=cmap, norm='log', rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    ##
    plt.xlabel("l [deg]")
    plt.xlim(180, -180)
    plt.ylabel("b [deg]")

    ax = plt.subplot(1, 3, 2)
    x = df['x']
    y = df['y']
    df_hist_xy, x_bins, y_bins = np.histogram2d(x, y, bins=bins)
    im = ax.pcolormesh(x_bins, y_bins, df_hist_xy.T, cmap=cmap, norm='log', rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [y_sun], '*', markersize=5, c='w')
    ##
    plt.xlabel("x [kpc]")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylabel("y [kpc]")
    if ylim is not None:
        ax.set_ylim(ylim)

    ax = plt.subplot(1, 3, 3)
    x = df['x']
    z = df['z']
    df_hist_xz, x_bins, z_bins = np.histogram2d(x, z, bins=bins)
    im = ax.pcolormesh(x_bins, z_bins, df_hist_xz.T,
                       cmap=cmap, norm='log', rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [z_sun], '*', markersize=5, c='w')
    ##
    plt.xlabel("x [kpc]")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylabel("z [kpc]")
    if ylim is not None:
        ax.set_ylim(zlim)



def plot_final_pane(dN, dQ, dom, xlim=None, ylim=None, zlim=None, bins=200, figsize=(9, 6), cmap='inferno'):

    x_sun = - Galactocentric().galcen_distance.to('kpc').value
    y_sun = 0.0
    z_sun = Galactocentric().z_sun.to('kpc').value

    plt.rc('font', size=6.0)
    plt.figure(figsize=figsize, layout='constrained')

    #
    # lb

    ax = plt.subplot(3, 3, 1)
    l = (dN['l'] + 180) % 360 - 180
    b = dN['b']
    dN_hist_lb, l_bins, b_bins = np.histogram2d(l, b, bins=bins)
    im = ax.pcolormesh(l_bins, b_bins, dN_hist_lb.T,
                       cmap=cmap, norm=LogNorm(), rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    plt.xlim(180, -180)
    plt.ylabel(r"$b$ [deg]")

    ax = plt.subplot(3, 3, 4)
    l = (dom.l + 180) % 360 - 180
    b = dom.b
    dQ_hist_lb, _, _ = np.histogram2d(l, b, bins=[l_bins, b_bins], weights=dQ.sum(axis=1))
    im = ax.pcolormesh(l_bins, b_bins, dQ_hist_lb.T,
                       cmap=cmap, norm=LogNorm(vmin=cb.vmin, vmax=cb.vmax), rasterized=True)
    plt.colorbar(im, ax=ax)
    plt.xlim(180, -180)
    plt.ylabel(r"$b$ [deg]")

    ax = plt.subplot(3, 3, 7)
    diff_lb = (dQ_hist_lb - dN_hist_lb) / (dQ_hist_lb + dN_hist_lb)
    im = ax.pcolormesh(l_bins, b_bins, diff_lb.T, cmap='bwr', rasterized=True)
    plt.colorbar(im, ax=ax)
    plt.xlim(180, -180)
    plt.xlabel(r"$l$ [deg]")
    plt.ylabel(r"$b$ [deg]")

    #
    # xy

    xy_edges = [bins, bins]
    if xlim is not None:
        xy_edges[0] = np.linspace(xlim[0], xlim[1], bins+1)
    if ylim is not None:
        xy_edges[1] = np.linspace(ylim[0], ylim[1], bins+1)

    ax = plt.subplot(3, 3, 2)
    x = dN['x']
    y = dN['y']
    dN_hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=xy_edges)
    im = ax.pcolormesh(x_edges, y_edges, dN_hist_xy.T,
                       cmap=cmap, norm=LogNorm(), rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [y_sun], '*', markersize=5, c='w')
    ax.set_ylabel(r"$y$ [kpc]")

    ax = plt.subplot(3, 3, 5)
    x = dom.xyz[0].ravel()
    y = dom.xyz[1].ravel()
    dQ_hist_xy, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=dQ.ravel())
    im = ax.pcolormesh(x_edges, y_edges, dQ_hist_xy.T,
                       cmap=cmap, norm=LogNorm(vmin=cb.vmin, vmax=cb.vmax), rasterized=True)
    plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [y_sun], '*', markersize=5, c='w')
    ax.set_ylabel(r"$y$ [kpc]")

    ax = plt.subplot(3, 3, 8)
    diff_xy = (dQ_hist_xy - dN_hist_xy) / (dQ_hist_xy + dN_hist_xy)
    im = ax.pcolormesh(x_edges, y_edges, diff_xy.T, cmap='bwr', rasterized=True)
    plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [y_sun], '*', markersize=5, c='w')
    ax.set_xlabel(r"$x$ [kpc]")
    ax.set_ylabel(r"$y$ [kpc]")

    #
    # xz

    xz_edges = [bins, bins]
    if xlim is not None:
        xz_edges[0] = np.linspace(xlim[0], xlim[1], bins+1)
    if zlim is not None:
        xz_edges[1] = np.linspace(zlim[0], zlim[1], bins+1)

    ax = plt.subplot(3, 3, 3)
    x = dN['x']
    z = dN['z']
    dN_hist_xz, x_edges, z_edges = np.histogram2d(x, z, bins=xz_edges)
    im = ax.pcolormesh(x_edges, z_edges, dN_hist_xz.T,
                       cmap=cmap, norm=LogNorm(), rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [z_sun], '*', markersize=5, c='w')
    ax.set_ylabel(r"$z$ [kpc]")

    ax = plt.subplot(3, 3, 6)
    x = dom.xyz[0].ravel()
    z = dom.xyz[2].ravel()
    dQ_hist_xz, _, _ = np.histogram2d(x, z, bins=(x_edges, z_edges), weights=dQ.ravel())
    im = ax.pcolormesh(x_edges, z_edges, dQ_hist_xz.T,
                       cmap=cmap, norm=LogNorm(vmin=cb.vmin, vmax=cb.vmax), rasterized=True)
    plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [z_sun], '*', markersize=5, c='w')
    ax.set_ylabel(r"$z$ [kpc]")

    ax = plt.subplot(3, 3, 9)
    diff_xz = (dQ_hist_xz - dN_hist_xz) / (dQ_hist_xz + dN_hist_xz)
    im = ax.pcolormesh(x_edges, z_edges, diff_xz.T, cmap='bwr', rasterized=True)
    plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [z_sun], '*', markersize=5, c='w')
    ax.set_xlabel(r"$x$ [kpc]")
    ax.set_ylabel(r"$z$ [kpc]")



def idw_interpolation(x, y, z, v, grid_x, grid_y, grid_z, eps=1e-12, k=4, p=2, workers=None):
    """
    Interpolate scalar values assigned to scattered points on a rectilinear grid using IDW.

    Parameters:
        x, y, z: 1D arrays of point coordinates (N,)
        v: 1D array of scalar values (N,)
        grid_x, grid_y, grid_z: 3D arrays of grid coordinates (from np.meshgrid)
        eps: a minimum distance assigned to avoid division by zero
    Parameters of `cKDTree.query` method:
        k: list of integers or integer, the number of nearest neighbors to use (default is 4)
        p: power parameter for IDW (default is 2)
        workers: number of workers to use for parallel processing; if -1 is given all CPU threads are used
    Returns:
        density: 3D array of estimated density on the grid
    """

    # Stack the input points
    points = np.vstack((x, y, z)).T
    tree = sp.spatial.cKDTree(points)

    # Flatten the grid for vectorized processing
    grid_points = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

    # Query k nearest neighbors for each grid point
    dists, idxs = tree.query(grid_points, k=k, p=p, workers=workers)

    # Avoid division by zero
    dists = np.maximum(dists, eps)
    weights = 1.0 / dists
    weights /= weights.sum(axis=1)[:, None]

    # Weighted sum of the values
    interp_values = np.sum(v[idxs] * weights, axis=1)

    # Reshape to grid
    return interp_values.reshape(grid_x.shape)



def idw_density(x, y, z, grid_x, grid_y, grid_z, radius, eps=1e-12, p=2, workers=None):
    """
    Estimate density of scattered points on a rectilinear grid using IDW.

    Parameters:
        x, y, z: 1D arrays of point coordinates (N,)
        grid_x, grid_y, grid_z: 3D arrays of grid coordinates (from np.meshgrid)
        radius: search radius for neighbors
        eps: a minimum distance assigned to avoid division by zero
    Parameters of `cKDTree.query` method:
        p: power parameter for IDW (default is 2)
        workers: number of workers to use for parallel processing; if -1 is given all CPU threads are used
    Returns:
        density: 3D array of estimated density on the grid
    """

    # Stack the input points
    points = np.vstack((x, y, z)).T
    tree = sp.spatial.cKDTree(points)

    # Flatten grid for vectorized processing
    grid_points = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

    # For each grid point, find all data points within 'radius'
    idxs = tree.query_ball_point(grid_points, r=radius)

    density = np.zeros(grid_points.shape[0])
    for i, neighbors in enumerate(idxs):
        if neighbors:
            dists = np.linalg.norm(points[neighbors] - grid_points[i], axis=1)
            # Avoid division by zero
            dists = np.maximum(dists, 1e-12)
            weights = 1.0 / dists**p
            density[i] = np.sum(weights)
        else:
            density[i] = 0.0  # No points nearby

    # Reshape to grid
    return density.reshape(grid_x.shape)



def plot_sky_projections(dom, N_lb, Q_lb, savefig_fname=None, show=False):

    fig, ax = fig_init(shape=(2, 2), dims=(8.0, 4.2), fontsize=5.0)
    fig.suptitle(f"Num. of sources per HEALPix {dom.hpxorder}")

    ax_ = ax[0,0]
    sky = np.full(dom.npix, np.nan)
    sky = N_lb
    plot_hpxmap(sky, vmin=1.0, interp=False, ax=ax_, title=r"$\mathtt{data}$")
    ax_.set_aspect('equal')

    ax_ = ax[0,1]
    sky = np.full(dom.npix, np.nan)
    sky = Q_lb
    plot_hpxmap(sky, vmin=1.0, interp=False, ax=ax_, title=r"$\mathtt{model}$")
    ax_.set_aspect('equal')

    ax_ = ax[1,0]
    sky = np.full(dom.npix, np.nan)
    sky = Q_lb - N_lb
    absmax = np.abs(sky).max()
    plot_hpxmap(sky, interp=False, norm='symlog', vmin=-absmax, vmax=absmax, cmap='bwr', ax=ax_,
                title=r"$\mathtt{model} - \mathtt{data}$")
    ax_.set_aspect('equal')

    ax_ = ax[1,1]
    mask = (Q_lb + N_lb) > 0.0
    sky = np.full(dom.npix, np.nan)
    sky[mask] = (Q_lb - N_lb)[mask] / (Q_lb + N_lb)[mask]
    plot_hpxmap(sky, interp=False, norm=None, vmin=-1.0, vmax=1.0, cmap='bwr', ax=ax_,
                title=r"$(\mathtt{model} - \mathtt{data})\,/\,(\mathtt{model} + \mathtt{data})$")
    ax_.set_aspect('equal')

    if savefig_fname:
        plt.savefig(savefig_fname)
    if show:  plt.show()
    plt.close()



def plot_cuts(dom, N_lbDG, Q_lbDG, savefig_fname=None, show=False):

    N_lbD = N_lbDG.sum(-1)
    N_lbG = N_lbDG.sum(-2)
    Q_lbD = Q_lbDG.sum(-1)
    Q_lbG = Q_lbDG.sum(-2)

    npix1 = int(np.sqrt(dom.npix/2))
    l_grid = np.linspace(-180.0, 180.0, 2*npix1)
    b_grid = np.linspace(-90.0, 90.0, npix1)
    b_mesh, l_mesh = np.meshgrid(b_grid, l_grid)

    hp = HEALPix(nside=dom.nside, order='nested', frame='icrs')
    co = SkyCoord(l_mesh, b_mesh, frame='galactic', unit='deg')
    hpx = hp.skycoord_to_healpix(co)


    plt.rc('font', size=5.5)
    plt.rc('axes', linewidth=0.6)
    plt.rc('legend', borderpad=0)
    plt.rc('legend', borderaxespad=0.25)
    plt.rc('lines', linewidth=0.75)
    inch = 2.54  ## cm
    width, height = 18/inch, 15.0/inch
    plt.figure(figsize=(width, height), layout='constrained')
    alpha = 0.7

    bs = [-30, -12, -5, 5, 20]
    ls = [-60, -10, 0, 10, 20]
    co = SkyCoord(ls, bs, frame='galactic', unit='deg')
    hp = HEALPix(nside=dom.nside, order='nested', frame='icrs')
    ipix_ = hp.skycoord_to_healpix(co)

    for k, b_ in enumerate(bs):
        plt.subplot(4, len(bs), 1+k)
        ib = np.argmin(b_grid < b_) - 1
        plt.step(l_grid, N_lbD[hpx[:,ib]].sum(axis=-1), where='mid', alpha=alpha, label=(r"$b={%g}^\circ$" % b_))
        plt.step(l_grid, Q_lbD[hpx[:,ib]].sum(axis=-1), c='k', where='mid', alpha=alpha)
        plt.legend(handlelength=0, loc='upper right', frameon=False)
        plt.xlabel(r"$l$ [deg]")
        plt.xlim(l_grid[-1], l_grid[0])
        plt.yscale('log')
        plt.ylim(ymin=1)

    for k, l_ in enumerate(ls):
        plt.subplot(4, len(bs), 1+len(bs)+k)
        il = np.argmin(l_grid < l_) - 1
        plt.step(b_grid, N_lbD[hpx[il,:]].sum(axis=-1), where='mid', alpha=alpha, label=(r"$l={%g}^\circ$" % l_))
        plt.step(b_grid, Q_lbD[hpx[il,:]].sum(axis=-1), c='k', where='mid', alpha=alpha)
        plt.legend(handlelength=0, loc='upper right', frameon=False)
        plt.xlabel(r"$b$ [deg]")
        plt.yscale('log')
        plt.ylim(ymin=1)

    for k, i in enumerate(ipix_):
        plt.subplot(4, len(bs), 1+2*len(bs)+k)
        plt.step(dom.D, N_lbD[i,:], where='mid', alpha=alpha, label=(r"$(l,b)=({%g}, {%g})^\circ$" % (ls[k], bs[k])))
        plt.step(dom.D, Q_lbD[i,:], c='k', where='mid', alpha=alpha)
        plt.legend(handlelength=0, loc='upper right', frameon=False)
        plt.xlabel(r"$D$ [kpc]")
        #plt.yscale('log')
        plt.ylim(ymin=1)

    for k, i in enumerate(ipix_):
        plt.subplot(4, len(bs), 1+3*len(bs)+k)
        plt.step(dom.G, N_lbG[i,:], where='mid', alpha=alpha, label=f"(l,b)=({ls[k]},{bs[k]})")
        plt.step(dom.G, Q_lbG[i,:], c='k', where='mid', alpha=alpha)
        plt.legend(handlelength=0, loc='upper right', frameon=False)
        plt.xlabel("G [mag]")
        plt.xlim(5.0, 17.0)
        #plt.yscale('log')
        plt.ylim(ymin=1)

    if savefig_fname:
        plt.savefig(savefig_fname)
    if show:  plt.show()
    plt.close()



def plot_projections(dom, df, Q_lbD, xlim=None, ylim=None, zlim=None, savefig_fname=None, show=False):

    x_sun = - Galactocentric().galcen_distance.to('kpc').value
    y_sun = 0.0
    z_sun = Galactocentric().z_sun.to('kpc').value

    plt.rc('font', size=6.0)
    plt.figure(figsize=(9, 6), layout='constrained')
    cmap = 'inferno'

    npix_b = int(np.sqrt(dom.npix/3.0))
    npix_x = 60

    #
    # lb

    ax = plt.subplot(3, 3, 1)
    l = (df['l'] + 180) % 360 - 180
    b = df['b']
    dN_hist_lb, l_bins, b_bins = np.histogram2d(l, b, bins=(2*npix_b, npix_b))
    im = ax.pcolormesh(l_bins, b_bins, dN_hist_lb.T,
                       cmap=cmap, norm=LogNorm(), rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    plt.xlim(180, -180)
    plt.ylabel(r"$b$ [deg]")

    #sky = np.full(dom.npix, np.nan)
    #sky = N_lb
    #plot_hpxmap(sky, vmin=1.0, interp=False, ax=ax_)

    ax = plt.subplot(3, 3, 4)
    l = (dom.l + 180) % 360 - 180
    b = dom.b
    dQ_hist_lb, _, _ = np.histogram2d(l, b, bins=[l_bins, b_bins], weights=Q_lbD.sum(axis=1))
    im = ax.pcolormesh(l_bins, b_bins, dQ_hist_lb.T,
                       cmap=cmap, norm=LogNorm(vmin=cb.vmin, vmax=cb.vmax), rasterized=True)
    plt.colorbar(im, ax=ax)
    plt.xlim(180, -180)
    plt.ylabel(r"$b$ [deg]")

    ax = plt.subplot(3, 3, 7)
    diff_lb = (dQ_hist_lb - dN_hist_lb) / (dQ_hist_lb + dN_hist_lb)
    im = ax.pcolormesh(l_bins, b_bins, diff_lb.T, cmap='bwr', rasterized=True)
    plt.colorbar(im, ax=ax)
    plt.xlim(180, -180)
    plt.xlabel(r"$l$ [deg]")
    plt.ylabel(r"$b$ [deg]")

    #
    # xy

    xy_edges = [npix_x, npix_x]
    if xlim is not None:
        xy_edges[0] = np.linspace(xlim[0], xlim[1], npix_x+1)
    if ylim is not None:
        xy_edges[1] = np.linspace(ylim[0], ylim[1], npix_x+1)

    ax = plt.subplot(3, 3, 2)
    x = df['x']
    y = df['y']
    dN_hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=xy_edges)
    im = ax.pcolormesh(x_edges, y_edges, dN_hist_xy.T,
                       cmap=cmap, norm=LogNorm(), rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [y_sun], '*', markersize=5, c='k')
    ax.set_ylabel(r"$y$ [kpc]")

    ax = plt.subplot(3, 3, 5)
    x = dom.xyz[0].ravel()
    y = dom.xyz[1].ravel()
    dQ_hist_xy, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=Q_lbD.ravel())
    im = ax.pcolormesh(x_edges, y_edges, dQ_hist_xy.T,
                       cmap=cmap, norm=LogNorm(vmin=cb.vmin, vmax=cb.vmax), rasterized=True)
    plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [y_sun], '*', markersize=5, c='k')
    ax.set_ylabel(r"$y$ [kpc]")

    ax = plt.subplot(3, 3, 8)
    diff_xy = (dQ_hist_xy - dN_hist_xy) / (dQ_hist_xy + dN_hist_xy)
    im = ax.pcolormesh(x_edges, y_edges, diff_xy.T, cmap='bwr', rasterized=True)
    plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [y_sun], '*', markersize=5, c='k')
    ax.set_xlabel(r"$x$ [kpc]")
    ax.set_ylabel(r"$y$ [kpc]")

    #
    # xz

    xz_edges = [npix_x, npix_x]
    if xlim is not None:
        xz_edges[0] = np.linspace(xlim[0], xlim[1], npix_x+1)
    if zlim is not None:
        xz_edges[1] = np.linspace(zlim[0], zlim[1], npix_x+1)

    ax = plt.subplot(3, 3, 3)
    x = df['x']
    z = df['z']
    dN_hist_xz, x_edges, z_edges = np.histogram2d(x, z, bins=xz_edges)
    im = ax.pcolormesh(x_edges, z_edges, dN_hist_xz.T,
                       cmap=cmap, norm=LogNorm(), rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [z_sun], '*', markersize=5, c='k')
    ax.set_ylabel(r"$z$ [kpc]")

    ax = plt.subplot(3, 3, 6)
    x = dom.xyz[0].ravel()
    z = dom.xyz[2].ravel()
    dQ_hist_xz, _, _ = np.histogram2d(x, z, bins=(x_edges, z_edges), weights=Q_lbD.ravel())
    im = ax.pcolormesh(x_edges, z_edges, dQ_hist_xz.T,
                       cmap=cmap, norm=LogNorm(vmin=cb.vmin, vmax=cb.vmax), rasterized=True)
    plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [z_sun], '*', markersize=5, c='k')
    ax.set_ylabel(r"$z$ [kpc]")

    ax = plt.subplot(3, 3, 9)
    diff_xz = (dQ_hist_xz - dN_hist_xz) / (dQ_hist_xz + dN_hist_xz)
    im = ax.pcolormesh(x_edges, z_edges, diff_xz.T, cmap='bwr', rasterized=True)
    plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [z_sun], '*', markersize=5, c='k')
    ax.set_xlabel(r"$x$ [kpc]")
    ax.set_ylabel(r"$z$ [kpc]")

    if savefig_fname:
        plt.savefig(savefig_fname)
    if show:  plt.show()
    plt.close()



def plot_xyz_residuals(dom, N_lbD, Q_lbD, norm='symlog', savefig_fname=None, show=False):
    x_sun, y_sun, z_sun = -8.2, 0.0, 0.21

    _, ax = fig_init(shape=(1, 3), dims=(8.0, 7.0), fontsize=5.0)
    cmap = 'bwr'

    xlim, ylim, zlim = (-20, 10), (-15, 15), (-15, 15)
    bins = 75
    x_edges = np.linspace(xlim[0], xlim[1], bins+1)
    y_edges = np.linspace(ylim[0], ylim[1], bins+1)
    z_edges = np.linspace(zlim[0], zlim[1], bins+1)

    field = (Q_lbD - N_lbD).ravel()

    ax_ = ax[0]
    ax_.set_title(r"$\mathtt{model} - \mathtt{data}$")
    x = dom.xyz[0].ravel()
    y = dom.xyz[1].ravel()
    field_xy, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=field)
    absmax = np.abs(field_xy).max()
    im = ax_.pcolormesh(x_edges, y_edges, field_xy.T, vmin=-absmax, vmax=absmax, cmap=cmap, norm=norm, rasterized=True)
    plt.colorbar(im, ax=ax_)
    ax_.plot([x_sun], [y_sun], '*', markersize=5, c='k')
    ax_.set_xlabel(r"$x$ [kpc]")
    ax_.set_ylabel(r"$y$ [kpc]")
    ax_.set_aspect('equal')

    ax_ = ax[1]
    ax_.set_title(r"$\mathtt{model} - \mathtt{data}$")
    x = dom.xyz[0].ravel()
    z = dom.xyz[2].ravel()
    field_xz, _, _ = np.histogram2d(x, z, bins=(x_edges, z_edges), weights=field)
    absmax = np.abs(field_xz).max()
    im = ax_.pcolormesh(x_edges, z_edges, field_xz.T, vmin=-absmax, vmax=absmax, cmap=cmap, norm=norm, rasterized=True)
    plt.colorbar(im, ax=ax_)
    ax_.plot([x_sun], [z_sun], '*', markersize=5, c='k')
    ax_.set_xlabel(r"$x$ [kpc]")
    ax_.set_ylabel(r"$z$ [kpc]")
    ax_.set_aspect('equal')

    ax_ = ax[2]
    ax_.set_title(r"$\mathtt{model} - \mathtt{data}$")
    y = dom.xyz[1].ravel()
    z = dom.xyz[2].ravel()
    field_yz, _, _ = np.histogram2d(y, z, bins=(y_edges, z_edges), weights=field)
    absmax = np.abs(field_yz).max()
    im = ax_.pcolormesh(y_edges, z_edges, field_yz.T, vmin=-absmax, vmax=absmax, cmap=cmap, norm=norm, rasterized=True)
    plt.colorbar(im, ax=ax_)
    ax_.plot([y_sun], [z_sun], '*', markersize=5, c='k')
    ax_.set_xlabel(r"$y$ [kpc]")
    ax_.invert_xaxis()
    ax_.set_ylabel(r"$z$ [kpc]")
    ax_.set_aspect('equal')

    if savefig_fname:
        plt.savefig(savefig_fname)
    if show:  plt.show()
    plt.close()
