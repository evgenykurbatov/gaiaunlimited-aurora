
import numpy as np
from numpy import pi, sqrt, exp, log, log10
import scipy as sp
from scipy.special import erfc
from scipy.interpolate import interp1d



class Rotation:
    """
    """

    def __init__(self, theta, eta, phi):
        self.theta, self.eta, self.phi = theta, eta, phi

        # Plane rotation matrix
        P = np.array([[ np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi)), 0],
                      [-np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi)), 0],
                      [0, 0, 1]])

        # Rodrigues rotation formula
        # https://stackoverflow.com/a/76703318/1038377
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        zetax = np.sqrt(1-eta**2) * np.cos(np.deg2rad(theta))
        zetay = np.sqrt(1-eta**2) * np.sin(np.deg2rad(theta))
        K = np.array([[0, 0, zetax], [0, 0, zetay], [-zetax, -zetay, 0]])
        #R = np.eye(3) + np.sqrt(1-eta**2) * K + (1+eta) * K.dot(K)
        R = np.eye(3) + np.sqrt(1-eta**2) * K + (1-eta) * K.dot(K)
        self.T = R.dot(P)



def lumfun_linear(M, eta):
    """
    Three-parameter LF
    """

    # Cumulative LF
    M_CLF = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
    CLF   = np.array([0.0, eta[2]*eta[1]*eta[0], eta[1]*eta[0], eta[0], 1.0])

    M_edges = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    LF = np.empty_like(M_edges)
    LF[0] = LF[-1] = 0.0
    LF[1:-1] = (CLF[1:] - CLF[:-1]) / (M_CLF[1:] - M_CLF[:-1])
    LF /= np.trapz(LF, M_edges)

    LF_interp = interp1d(M_edges, LF, bounds_error=False, fill_value=(0.0, 0.0))

    return LF_interp(M)



class ZeroDensity:
    """Zero density everywhere
    """

    def __call__(self, x, y, z, *args, **kwargs):
        return np.zeros_like(x)



class Density_Lane23_AD_SC(Rotation):
    """Density profile

    Lane, Bovy & Mackereth. The stellar mass of the Gaia-Sausage/Enceladus accretion remnant (2023)
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1209L

    'AD' selection, 'SC' model.
    """

    def __init__(self):
        self.alpha = -0.57
        self.r_1 = 8.57
        self.p = 0.54
        self.q = 0.46

        super().__init__(theta=147.0, eta=0.84, phi=99.5)

    def __call__(self, x, y, z, *args, **kwargs):
        r_ = np.einsum('ij,j...->i...', self.T, np.array([x, y, z]))
        m = np.sqrt(r_[0]**2 + (r_[1]/self.p)**2 + (r_[2]/self.q)**2)
        rho = m**(-self.alpha) * np.exp(-m/self.r_1)
        return rho



class Density_Lane23_AD_BPL(Rotation):
    """Density profile

    Lane, Bovy & Mackereth. The stellar mass of the Gaia-Sausage/Enceladus accretion remnant (2023)
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1209L

    'AD' selection, 'BPL' model.
    """

    def __init__(self):
        self.alpha_1 = 1.05
        self.alpha_2 = 3.79
        self.r_1 = 23.59
        self.p = 0.55
        self.q = 0.47

        super().__init__(theta=146.6, eta=0.84, phi=98.5)

    def __call__(self, x, y, z, *args, **kwargs):
        r_ = np.einsum('ij,j...->i...', self.T, np.array([x, y, z]))
        m = np.sqrt(r_[0]**2 + (r_[1]/self.p)**2 + (r_[2]/self.q)**2)
        rho = np.where(m <= self.r_1, \
                       m**(-self.alpha_1), \
                       self.r_1**(self.alpha_2-self.alpha_1) * m**(-self.alpha_2))
        return rho



class Density_Lane23_ELZ_SCDn(Rotation):
    """Density profile

    Lane, Bovy & Mackereth. The stellar mass of the Gaia-Sausage/Enceladus accretion remnant (2023)
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1209L

    Mackereth et al. The age-metallicity structure of the Milky Way disc using APOGEE (2017)
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1209L

    'e-Lz' selection, 'SC+D' model.

    NB: THE DISC IS NULLYFIED (that the 'n' is standing for)
    """

    def __init__(self):
        self.alpha = -3.22
        self.r_1 = 3.46
        self.p = 0.58
        self.q = 0.36
        self.R_sol = 8.2
        super().__init__(theta=78.8, eta=0.99, phi=99.6)

    def _rho_body(self, x, y, z, *args, **kwargs):
        r_ = np.einsum('ij,j...->i...', self.T, np.array([x, y, z]))
        m = np.sqrt(r_[0]**2 + (r_[1]/self.p)**2 + (r_[2]/self.q)**2)
        rho = m**(-self.alpha) * np.exp(-m/self.r_1)
        return rho

    def __call__(self, x, y, z, *args, **kwargs):
        rho_body = self._rho_body(x, y, z) / self._rho_body(self.R_sol, 0, 0)
        rho = rho_body
        return rho



class Density_Lane23_ELZ_DBPLDn(Rotation):
    """Density profile

    Lane, Bovy & Mackereth. The stellar mass of the Gaia-Sausage/Enceladus accretion remnant (2023)
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1209L

    Mackereth et al. The age-metallicity structure of the Milky Way disc using APOGEE (2017)
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1209L

    'e-Lz' selection, 'DBPL+D' model.

    NB: THE DISC IS NULLYFIED (that the 'n' is standing for)
    """

    def __init__(self):
        self.alpha_1 = 0.64
        self.alpha_2 = 4.08
        self.alpha_3 = 6.8
        self.r_1 = 17.4
        self.r_2 = 34.17
        self.p = 0.61
        self.q = 0.37
        #self.f_disk = 0.55
        self.R_sol = 8.12
        super().__init__(theta=77.7, eta=0.99, phi=98.2)

    def _rho_body(self, x, y, z):
        r_ = np.einsum('ij,j...->i...', self.T, np.array([x, y, z]))
        m = np.sqrt(r_[0]**2 + (r_[1]/self.p)**2 + (r_[2]/self.q)**2)
        rho = np.where(m <= self.r_1, \
                       m**(-self.alpha_1), \
                       np.where(m <= self.r_2,
                                self.r_1**(self.alpha_2-self.alpha_1) * m**(-self.alpha_2),
                                self.r_1**(self.alpha_2-self.alpha_1) * self.r_2**(self.alpha_3-self.alpha_2) * m**(-self.alpha_3)))
        return rho

    def __call__(self, x, y, z, *args, **kwargs):
        rho_body = self._rho_body(x, y, z) / self._rho_body(self.R_sol, 0, 0)
        rho = rho_body
        return rho



class Density_Lane23_JRLZ_SPLDn(Rotation):
    """Density profile

    Lane, Bovy & Mackereth. The stellar mass of the Gaia-Sausage/Enceladus accretion remnant (2023)
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1209L

    Mackereth et al. The age-metallicity structure of the Milky Way disc using APOGEE (2017)
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1209L

    'sqrt{J_R}-Lz' selection, 'SPL+D' model.

    NB: THE DISC IS NULLYFIED (that the 'n' is standing for)
    """

    def __init__(self):
        self.alpha = 2.39
        self.p = 0.57
        self.q = 0.69
        self.R_sol = 8.2
        super().__init__(theta=167.78, eta=0.93, phi=105.4)

    def _rho_body(self, x, y, z, *args, **kwargs):
        r_ = np.einsum('ij,j...->i...', self.T, np.array([x, y, z]))
        m = np.sqrt(r_[0]**2 + (r_[1]/self.p)**2 + (r_[2]/self.q)**2)
        rho = m**(-self.alpha)
        return rho

    def __call__(self, x, y, z, *args, **kwargs):
        rho_body = self._rho_body(x, y, z) / self._rho_body(self.R_sol, 0, 0)
        rho = rho_body
        return rho



class LossBase:

    def __init__(self, xyz, density_gse, keys, aux):
        """
        """

        # Dims: 3, hpx, D'
        self.xyz = xyz

        # Pre-calculate the GSE density values for the domain and the sample
        # Dims: hpx, D'
        self.rho_gse = density_gse(self.xyz[0], self.xyz[1], self.xyz[2])

        # Keys to convert a numerical array to a dictionary
        self.keys = keys

        # Selection function
        # Dims: hpx, G, D'
        self.TF = aux['TF']
        # Design matrix (transfer function)
        # Absolute magnitudes
        # Dims: hpx, D', G
        self.MG = aux['MG']
        # Bins in apparent magnitudes
        self.dG = aux['dG']
        # Volume elements
        self.dVol = aux['dVol']


    def pmf_lbDG(self, theta, low, high, lumfun):

        # Check the bounds
        if np.any(theta < low) | np.any(theta > high):
            return np.zeros((self.xyz.shape[1], self.dG.size))

        # Convert the `theta` array to a dictionary
        params = dict(zip(self.keys, theta))

        # Dims: 3, hpx, D'
        xyz = self.xyz

        # Dims: hpx, D'
        rho_au  = self.density_au(xyz[0], xyz[1], xyz[2], params)
        xi = params['xi']
        rho = (1-xi)*rho_au + xi*self.rho_gse
        # This will be PMF over the (hpx, D') space
        # Dims: hpx, D'
        q = rho * self.dVol
        q /= q.sum()

        # Dims: hpx, G, D'
        Phi = lumfun(self.MG, (params['lf1'], params['lf2'], params['lf3']))
        # This will be PMF over the G space
        dPsi_ = np.einsum('...kl,k->...kl', Phi, self.dG)
        # Normalize over G
        # Dims: hpx, D'
        norm = dPsi_.sum(axis=-2)
        mask = norm > 0.0
        norm_inv = np.zeros_like(norm)
        norm_inv[mask] = 1./norm[mask]
        # Note: there might be no 'hpx' index
        # Dims: hpx, G, D'
        dPsi = np.einsum('...kl,...l->...kl', dPsi_, norm_inv)

        # PMF in the observables' space, after selection applied
        # Dims: hpx, G
        # Note: there might be no 'hpx' index
        p = np.einsum('...jkl,...l,...kl->...jk', self.TF, q, dPsi)
        p /= p.sum()

        return p


    def log_prob(self, theta, low, high, lumfun, N_lbDG):

        # Check the bounds
        if np.any(theta < low) | np.any(theta > high):
            return -np.inf

        p = self.pmf_lbDG(theta, low, high, lumfun)
        mask = p > 0.0

        # KL-like likelihood, which is the multinomial when all redundants are dropped off
        lnL = np.sum(N_lbDG[mask] * np.log(p[mask]))

        return lnL


    def log_loss(self, theta, low, high, lumfun, N_lbDG):
        return -self.log_prob(theta, low, high, lumfun, N_lbDG)



class Loss_AuPL(LossBase):

    def __init__(self, xyz, density_gse, keys, aux):
        super().__init__(xyz, density_gse, keys, aux)


    def density_au(self, x, y, z, params):
        b, s, k = params['b'], params['s'], params['k']
        m2 = (x**2 + y**2)/(b*s)**2 + z**2/b**2
        rho = 1.0 / (1.0 + m2)**(0.5*k)
        return rho



class Loss_AuPL_ED(LossBase):
    """Flattened Aurora profile + exponential disk contamination
    """

    def __init__(self, xyz, density_gse, keys, aux):
        super().__init__(xyz, density_gse, keys, aux)


    def density_au(self, x, y, z, params):
        xid = params['xid']
        rho_au = self.density_au0(x, y, z, params)
        rho_d  = self.density_disc(x, y, z, params)
        rho = (1.0-xid)*rho_au + xid*rho_d
        return rho


    def density_au0(self, x, y, z, params):
        b, s, k = params['b'], params['s'], params['k']
        m2 = (x**2 + y**2)/(b*s)**2 + z**2/b**2
        rho_au = 1.0 / (1.0 + m2)**(0.5*k)
        return rho_au


    def density_disc(self, x, y, z, params):
        rd, zd = params['rd'], params['zd']
        rho_d = np.exp(- np.sqrt(x**2 + y**2)/rd - np.abs(z)/zd)
        return rho_d



class Loss_AuPL_PD(LossBase):
    """Flattened Aurora profile + power-law disk contamination
    """

    def __init__(self, xyz, density_gse, keys, aux):
        super().__init__(xyz, density_gse, keys, aux)


    def density_au(self, x, y, z, params):
        xid = params['xid']
        rho_au = self.density_au0(x, y, z, params)
        rho_d  = self.density_disc(x, y, z, params)
        rho = (1.0-xid)*rho_au + xid*rho_d
        return rho


    def density_au0(self, x, y, z, params):
        b, s, k = params['b'], params['s'], params['k']
        m2 = (x**2 + y**2)/(b*s)**2 + z**2/b**2
        rho_au = 1.0 / (1.0 + m2)**(0.5*k)
        return rho_au


    def density_disc(self, x, y, z, params):
        bd, sd, kd = params['bd'], params['sd'], params['kd']
        m2 = (x**2 + y**2)/(bd*sd)**2 + z**2/bd**2
        rho_d = 1.0 / (1.0 + m2)**(0.5*kd)
        return rho_d
