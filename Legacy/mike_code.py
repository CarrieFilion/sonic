import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import eval_genlaguerre
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import eval_genlaguerre
import time

class LaguerreAmplitudes:
    """
    LaguerreAmplitudes class for calculating Laguerre basis amplitudes.
    This class provides methods for calculating Laguerre basis amplitudes based on Weinberg & Petersen (2021).
    Parameters:
        rscl (float): Scale parameter for the Laguerre basis.
        mass (array-like): Mass values for particles.
        phi (array-like): Angular phi values.
        velocity (array-like): Velocity values.
        R (array-like): Radial values.
        mmax (int): Maximum order parameter for m.
        nmax (int): Maximum order parameter for n.
    Methods:
        gamma_n(nrange, rscl): Calculate the Laguerre alpha=1 normalisation.
        G_n(R, nrange, rscl): Calculate the Laguerre basis.
        n_m(): Calculate the angular normalisation.
        laguerre_amplitudes(): Calculate Laguerre amplitudes for the given parameters.
        laguerre_reconstruction(rr, pp): Calculate Laguerre reconstruction.
    Attributes:
        rscl (float): Scale parameter for the Laguerre basis.
        mass (array-like): Mass values for particles.
        phi (array-like): Angular phi values.
        velocity (array-like): Velocity values.
        R (array-like): Radial values.
        mmax (int): Maximum order parameter for m.
        nmax (int): Maximum order parameter for n.
        coscoefs (array-like): Cosine coefficients.
        sincoefs (array-like): Sine coefficients.
        reconstruction (array-like): Laguerre reconstruction result.
    """

    def __init__(self, rscl, mmax, nmax, R, phi, mass=1., velocity=1.):
        """
        Initialize the LaguerreAmplitudes instance with parameters.
        Args:
            rscl (float): Scale parameter for the Laguerre basis.
            mmax (int): Maximum Fourier harmonic order.
            nmax (int): Maximum Laguerre order.
            R (array-like): Radial values.
            velocity (array-like): Velocity values.
            mass (integer or array-like): Mass values for particles.
            phi (integer or array-like): Angular phi values.
        """
        self.rscl = rscl
        self.mmax = mmax
        self.nmax = nmax
        self.R = R
        self.phi = phi
        self.mass = mass
        self.velocity = velocity


        # run the amplitude calculation
        self.laguerre_amplitudes()

    def _gamma_n(self,nrange, rscl):
        """
        Calculate the Laguerre alpha=1 normalisation.
        Args:
            nrange (array-like): Range of order parameters.
            rscl (float): Scale parameter for the Laguerre basis.
        Returns:
            array-like: Laguerre alpha=1 normalisation values.
        """
        return (rscl / 2.) * np.sqrt(nrange + 1.)

    def _G_n(self,R, nrange, rscl):
        """
        Calculate the Laguerre basis.
        Args:
            R (array-like): Radial values.
            nrange (array-like): Range of order parameters.
            rscl (float): Scale parameter for the Laguerre basis.
        Returns:
            array-like: Laguerre basis values.
        """
        laguerrevalues = np.array([eval_genlaguerre(n, 1, 2 * R / rscl)/self._gamma_n(n, rscl) for n in nrange])
        return np.exp(-R / rscl) * laguerrevalues

    def _n_m(self):
        """
        Calculate the angular normalisation.
        Returns:
            array-like: Angular normalisation values.
        """
        deltam0 = np.zeros(self.mmax)
        deltam0[0] = 1.0
        return np.power((deltam0 + 1) * np.pi / 2., -0.5)

    def laguerre_amplitudes(self):
        """
        Calculate Laguerre amplitudes for the given parameters.
        Returns:
            tuple: Tuple containing the cosine and sine amplitudes.
        """

        G_j = self._G_n(self.R, np.arange(0,self.nmax,1), self.rscl)

        nmvals = self._n_m()
        cosm = np.array([nmvals[m]*np.cos(m*self.phi) for m in np.arange(0,self.mmax,1)])
        sinm = np.array([nmvals[m]*np.sin(m*self.phi) for m in np.arange(0,self.mmax,1)])

        # broadcast to sum values
        self.coscoefs = np.nansum(cosm[:, np.newaxis, :] * G_j[np.newaxis, :, :] * self.mass * self.velocity,axis=2)
        self.sincoefs = np.nansum(sinm[:, np.newaxis, :] * G_j[np.newaxis, :, :] * self.mass * self.velocity,axis=2)
   
    def laguerre_amplitudes_returns(self):
        """
        Calculate Laguerre amplitudes for the given parameters.
        Returns:
            tuple: Tuple containing the cosine and sine amplitudes.
        modified to actually return tuple
        """

        G_j = self._G_n(self.R, np.arange(0,self.nmax,1), self.rscl)

        nmvals = self._n_m()
        cosm = np.array([nmvals[m]*np.cos(m*self.phi) for m in np.arange(0,self.mmax,1)])
        sinm = np.array([nmvals[m]*np.sin(m*self.phi) for m in np.arange(0,self.mmax,1)])

        # broadcast to sum values
        coscoefs = np.nansum(cosm[:, np.newaxis, :] * G_j[np.newaxis, :, :] * self.mass * self.velocity,axis=2)
        sincoefs = np.nansum(sinm[:, np.newaxis, :] * G_j[np.newaxis, :, :] * self.mass * self.velocity,axis=2)
        print('returning coscoefs, sincoefs')
        return coscoefs, sincoefs
        
    def laguerre_reconstruction(self, rr, pp):
        """
        Reconstruct a function using Laguerre amplitudes.
        Args:
            rr (array-like): Radial values.
            pp (array-like): Angular phi values.
        This method reconstructs a function using the Laguerre amplitudes calculated with the `laguerre_amplitudes` method.
        Returns:
            array-like: The reconstructed function values.
        """
        nmvals = self._n_m()
        G_j = self._G_n(rr, np.arange(0, self.nmax, 1), self.rscl)

        fftotal = 0.
        for m in range(0, self.mmax):
            for n in range(0, self.nmax):
                fftotal += self.coscoefs[m, n] * nmvals[m] * np.cos(m * pp) * G_j[n]
                fftotal += self.sincoefs[m, n] * nmvals[m] * np.sin(m * pp) * G_j[n]

        self.reconstruction = 0.5 * fftotal
        


