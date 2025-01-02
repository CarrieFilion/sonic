# Import necessary Python libraries and packages

import numpy as np
import time
import warnings
import csv
import os
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

from scipy.special import eval_genlaguerre
from scipy.optimize import curve_fit
from scipy import stats

class ExpandGalaxies:

    def __init__(self, filename, rscl, mmax, nmax, R=None, phi=None, mass=None, velocity=1.0):
        """
        A class to extract and process galaxy data from images, do Fourier Laguerre expansion

        Parameters:
        ----------
        filename : str
            path to image
        rscl : float
            Scaling parameter for Laguerre functions.
        mmax : int
            Maximum order of the Laguerre expansion.
        nmax : int
            Maximum radial quantum number of the Laguerre expansion.
        R : array-like, optional
            Radial coordinates of pixels.
        phi : array-like, optional
            Angular coordinates of pixels.
        mass : array-like, optional
            Mass or intensity values of pixels.
        velocity : float, optional
            Velocity parameter used in Laguerre amplitude calculation.

        """
        # Filename for image
        self.filename = filename
        
        self.rscl = rscl
        self.mmax = mmax
        self.nmax = nmax
        self.R = R
        self.phi = phi
        self.mass = mass
        self.velocity = velocity

        if self.R is not None and self.phi is not None:
            self.laguerre_amplitudes()

        


    def sersic_profile(self, r, I0, Reff, n):
        """
        Calculate the Sersic profile for galaxy luminosity distribution.

        Parameters:
        ----------
        r : float or array_like
            Radial distance from the galactic center.
        I0 : float
            Central surface brightness.
        Reff : float
            Effective galaxy radius.
        n : float
            Sersic index number.

        Returns:
        -------
        float or array_like
            Luminosity distribution at given radial distance from galactic centre.
        """
        return I0 * np.exp(- (r / Reff)**(1/n))

    def determine_galaxy_radius(self, image_data, n_guess=1.0):
        """
        Determine the main galaxy radius beyond which background galaxies are masked.

        Parameters:
        ----------
        image_data : ndarray
            Image data of the galaxy.
        n_guess : float, optional
            Initial guess for the Sersic index (default is 1.0 as they approximates to an exponential profile).

        Returns:
        -------
        float
            Radius of the main galaxy beyond which background galaxies are masked.
        """
        
        # Calculate the cutout center and image dimensions
        xdim, ydim = image_data.shape
        x_center, y_center = xdim / 2, ydim / 2
        
        # Define a radial distance array from the galactic center
        x_indices, y_indices = np.indices(image_data.shape)
        radial_distances = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
        
        # Flatten image data and radial distance array created above
        intensity_values = image_data.flatten()
        radial_distances_flat = radial_distances.flatten()
        
        # Remove NaN values as they can cause an error
        mask = ~np.isnan(intensity_values)
        intensity_values = intensity_values[mask]
        radial_distances_flat = radial_distances_flat[mask]
        
        # Sort intensity values by radial distance
        sorted_indices = np.argsort(radial_distances_flat)
        sorted_distances = radial_distances_flat[sorted_indices]
        sorted_intensities = intensity_values[sorted_indices]
        
        # If loop to deal with cases where there aren't enough points for scipy.optimize curvefit method to work on
        if len(sorted_distances) < 10:
            print("Not enough data points for fitting.")
            return np.nan
        
        # Try and except block to handle possible errors in Sersic Profile fitting
        try:
            
            # Perform scipy curve fitting using scipy.optimize. Disc galaxies here have exponential profile ie n ~ 1
            #I've added bounds, edited p0 and maxfev to try to speed this up
            popt, _ = curve_fit(self.sersic_profile, sorted_distances, sorted_intensities, p0=(np.mean(sorted_intensities), \
                                np.mean(sorted_distances), 1.0), maxfev=1000, bounds = ([0, 0, -5], [99999999, 99999999, 999999]))
            
            # Assign values to the result got above
            I0, Reff, n = popt
            
            # Define the galaxy radius as a multiple of effective radius
            galaxy_radius = 10 * Reff 
            print('Reff = ', Reff, ' Galaxy Radius: ', galaxy_radius)
            return galaxy_radius
        
        except Exception as e:
            print(f"Error fitting Sersic profile: {str(e)}")
            return np.nan

    def get_image_data(self, galaxy_radius=None):
        """
        Obtain image array of the main galaxy while masking out background galaxies.

        Parameters:
        ----------
        filename : str
            Path to the galaxy image
            
        galaxy_radius : float, optional
            Radius of the galaxy beyond which background galaxies are masked (default is None).

        Returns:
        -------
        ndarray
            Cropped image data of the galaxy.
        """
        
        # Open FITS file
        try: 
            image = np.asarray(Image.open(self.filename))
        except:
            print('uhoh - no image to open, try a different path')
        av_image = np.mean(image, axis=2).T #combine color channels, transpose bc images go rows, columns, colors

            
        # If galaxy radius isn't fixed find it. 
        if galaxy_radius is None:
            self.galaxy_radius = self.determine_galaxy_radius(av_image)

        ###########
        ############ RETURN TO THIS BIT
        # Calculate radial distance from galactic center, assuming to be middle of image
        x_indices, y_indices = np.indices(av_image.shape)
        distance_from_center = np.sqrt((x_indices - (len(x_indices)/2))**2 + (y_indices - (len(y_indices)/2))**2)
            
        # Mask high pixel intensity spots only outside a certain radius
        mask_galaxy = distance_from_center <= self.galaxy_radius
        
        # Define a threshold for background galaxy masking using sigma clipping
        metric_threshold = np.median(av_image[mask_galaxy]) + 3 * stats.median_abs_deviation(av_image[mask_galaxy])
        
        # Define threshold where if the pixel value (outside the galaxy radius) exceeds metric_threshold it is blanked
        mask_background = av_image > metric_threshold
        
        # Combine galaxy and background masks to blank out regions. '~' reverses the conditions of mask_galaxy
        mask_to_blank = mask_background & ~mask_galaxy
        
        # Apply NaNs to blank out regions in cropped_data and cropped_uncertainty
        av_image[mask_to_blank] = np.nan

        return av_image



    def readsnapshot(self):
        """
        Read and preprocess image data from an HDF5 file.

        Parameters:
        -----------
        filename 

        Returns:
        --------
        rr : ndarray, shape [x, y]
            Radial coordinates of pixels. 
        pp : ndarray, shape [x, y]
            Angular coordinates of pixels.
        xpix : ndarray
            Meshgrid of x-pixel coordinates.
        ypix : ndarray
            Meshgrid of y-pixel coordinates.
        image_array : ndarray
            Processed image data.
        xdim : int
            Dimension of x-axis in the image array.
        ydim : int
            Dimension of y-axis in the image array.

        Notes:
        ------
        This method flattens the image data and applies a mask to exclude pixels based on specific metric.
        """ 
        #get masked image data
        image_array = self.get_image_data(self.filename)
        # Get the shape of image_array (2D shape)
        xdim, ydim = image_array.shape
        
        # Recreate xpixels and ypixels and set bounds
        xpixels = np.linspace(-xdim / 2, xdim / 2, xdim) #middle should be at 0, 0
        ypixels = np.linspace(-ydim / 2, ydim / 2, ydim)
        
        # Create meshgrid and important to include the indexing here
        self.xpix, self.ypix = np.meshgrid(xpixels, ypixels, indexing='ij')
    
        # Set the radial and phi values
        rr, pp = np.sqrt(self.xpix**2 + self.ypix**2), np.arctan2(self.ypix, self.xpix)
        
        # Here mask excludes negative pixel contributions along with those outside 3x the galaxy radius
        gvals = np.where((rr > self.galaxy_radius) | (image_array < 0))
        
        # Apply the mask by turning out of bound values to nan
        rr[gvals], pp[gvals], image_array[gvals] = np.nan, np.nan, np.nan
        self.R = rr.flatten().copy()
        self.phi = pp.flatten().copy()
        self.mass = image_array.flatten().copy()
    
        # Returning xdim and ydim here also as they are used when plotting the contour maps
        return rr, pp, self.xpix, self.ypix, image_array, xdim, ydim

    def _gamma_n(self, nrange, rscl):
        """
        Compute the normalization constant 'gamma_n' for Laguerre functions.

        Parameters:
        -----------
        nrange : array-like
            Range of modal numbers for Laguerre functions.
        rscl : float
            Scale Length for Laguerre functions.

        Returns:
        --------
        gamma_n : ndarray
            Normalization constants for Laguerre functions.

        Notes:
        ------
        This function is used internally within LaguerreAmplitudes class methods.
        """
        return (rscl / 2.) * np.sqrt(nrange + 1.)

    def _G_n(self, R, nrange, rscl):
        """
        Calculate the Laguerre basis functions G_n(R).

        Parameters:
        -----------
        R : array-like
            Radial coordinates.
        nrange : array-like
            Range of modal numbers for Laguerre functions.
        rscl : float
            Scale Length for Laguerre functions.

        Returns:
        --------
        G_n : ndarray
            Laguerre basis functions evaluated at radial coordinates R.

        Notes:
        ------
        This function is used internally within LaguerreAmplitudes class methods.
        """
        laguerrevalues = np.array([eval_genlaguerre(n, 1, 2 * R / rscl) / self._gamma_n(n, rscl) for n in nrange])
        return np.exp(- R / rscl) * laguerrevalues

    def _n_m(self):
        """
        Compute the angular momentum normalization coefficients.

        Returns:
        --------
        nmvals : ndarray
            Normalization coefficients for angular momentum.

        Notes:
        ------
        This function is used internally within LaguerreAmplitudes class methods.
        """
        deltam0 = np.zeros(self.mmax)
        deltam0[0] = 1.0
        return np.power((deltam0 + 1) * np.pi / 2., -0.5)

    def laguerre_amplitudes(self):
        """
        Compute the Laguerre coefficients (coscoefs and sincoefs) for the given parameters.

        Notes:
        ------
        This method calculates the coefficients using the current values of R, phi, mass,
        and velocity attributes of the object.
        """
        G_j = self._G_n(self.R, np.arange(0, self.nmax), self.rscl)
        nmvals = self._n_m()
        cosm = np.array([nmvals[m] * np.cos(m * self.phi) for m in range(self.mmax)])
        sinm = np.array([nmvals[m] * np.sin(m * self.phi) for m in range(self.mmax)])

        self.coscoefs = np.nansum(cosm[:, np.newaxis, :] * G_j[np.newaxis, :, :] * self.mass * self.velocity, axis=2)
        self.sincoefs = np.nansum(sinm[:, np.newaxis, :] * G_j[np.newaxis, :, :] * self.mass * self.velocity, axis=2)

    def calculate_A1(self, n_min=0, n_max=None):
        """
        Calculate the A1 matrix, which is used for center finding.

        Parameters:
        -----------
        n_min : int, optional
            Minimum mode number for Laguerre functions to include in calculation.
        n_max : int, optional
            Maximum mode number for Laguerre functions to include in calculation.

        Returns:
        --------
        A1 : float
            A1 matrix value, used for center finding.

        Notes:
        ------
        This method calculates A1 based on the specified range of Laguerre mode numbers.
        """
        if n_max is None:
            n_max = self.nmax
            
        # Begin by calculating cos and sine values for range of n values at m = 1. Returns  A1 expression
        c1n = self.coscoefs[1, n_min:n_max]
        s1n = self.sincoefs[1, n_min:n_max]

        return np.sqrt(np.sum(c1n**2 + s1n**2))

    def calculate_gradient(self, x, y, step=1e-3, n_min=0, n_max=None):
        """
        Calculate the gradient of A1 matrix at given coordinates (x, y) using finite differences.

        Parameters:
        -----------
        x : float
            x-coordinate for which gradient is calculated.
        y : float
            y-coordinate for which gradient is calculated.
        step : float, optional
            Step size for finite difference calculation.
        n_min : int, optional
            Minimum modal number for Laguerre functions to include in calculation.
        n_max : int, optional
            Maximum modal number for Laguerre functions to include in calculation.

        Returns:
        --------
        gradient : ndarray
            Gradient of A1 matrix at coordinates (x, y).

        Notes:
        ------
        This method uses finite differences to approximate the gradient of A1 matrix
        at the specified coordinates (x, y).
        """
        original_R, original_phi = self.R.copy(), self.phi.copy()
        
        
        def compute_A1_at_shift(shift_x, shift_y):
            
            self.R = np.sqrt((self.xpix - shift_x)**2 + (self.ypix - shift_y)**2).flatten()
            self.phi = np.arctan2(self.ypix - shift_y, self.xpix - shift_x).flatten()
            self.laguerre_amplitudes()
            return self.calculate_A1(n_min, n_max)

        A1_x_plus_step = compute_A1_at_shift(x + step, y)
        A1_x_minus_step = compute_A1_at_shift(x - step, y)
        A1_y_plus_step = compute_A1_at_shift(x, y + step)
        A1_y_minus_step = compute_A1_at_shift(x, y - step)

        # Reset to original values
        self.R, self.phi = original_R, original_phi
        self.laguerre_amplitudes()

        # Compute the gradient. Note: Denominator here is 2 * step size
        dA1_dx = (A1_x_plus_step - A1_x_minus_step) / (2 * step)
        dA1_dy = (A1_y_plus_step - A1_y_minus_step) / (2 * step)

        return np.array([dA1_dx, dA1_dy])

    def find_center(self, initial_guess=(0, 0), tol=1e-2, max_iter=1000, n_min=0, n_max=None):
        #lowered max_iter for timing
        """
        Find the center of the Laguerre expansion using Newton-Raphson method.

        Parameters:
        -----------
        initial_guess : tuple, optional
            Initial guess for center coordinates (x, y).
        tol : float, optional
            Tolerance for convergence of Newton-Raphson method.
        max_iter : int, optional
            Maximum number of iterations for Newton-Raphson method.
        n_min : int, optional
            Minimum modal number for Laguerre functions to include in calculation.
        n_max : int, optional
            Maximum modal number for Laguerre functions to include in calculation.

        Returns:
        --------
        x : float
            x-coordinate of the estimated center.
        y : float
            y-coordinate of the estimated center.
        reconstruction : ndarray
            Reconstructed image using Laguerre coefficients after center finding.

        Notes:
        ------
        This method iteratively applies Newton-Raphson method to refine the center
        estimate until tolerance limit is met.
        """
        
        # Initial guess for centre seperated into its coordinates
        x, y = initial_guess

        # Create a loop limited by a certain max number of iterations
        for i in range(max_iter):
            if i % 50 == 0:
                print('center finding iteration', i)
            # Calculate the gradient for this guess for a standardised nmin and nmax
            gradient = self.calculate_gradient(x, y, n_min=n_min, n_max=n_max)
            step_direction = - gradient
            step_size = 1e-3 #
            
            # Update the initial guess
            x += step_size * step_direction[0]
            y += step_size * step_direction[1]

            # Break the loop only when its norm of negative gradient is < tolerance set
            if np.linalg.norm(step_direction) < tol:
                break

        # After finding the center, recalculate and set R and phi
        self.R = np.sqrt((self.xpix - x)**2 + (self.ypix - y)**2).flatten()
        self.phi = np.arctan2(self.ypix - y, self.xpix - x).flatten()
        
        # Recalculate the Laguerre amplitudes
        self.laguerre_amplitudes()

        # Perform the updated reconstruction
        reconstruction = self.laguerre_reconstruction(np.sqrt((self.xpix - x)**2 + (self.ypix - y)**2), \
                                                      np.arctan2(self.ypix - y, self.xpix - x))

        return x, y, reconstruction

    def find_center_moments(self):
        """
        Calculate the center of mass (centroid) of the image's surface mass distribution.

        Returns:
        --------
        x_mass : float
            x-coordinate of the centroid.
        y_mass : float
            y-coordinate of the centroid.

        Notes:
        ------
        This method calculates the center of mass based on the mass distribution
        represented by the image mass array.
        """

        x_mass = np.nansum(self.xpix.flatten() * self.mass) / np.nansum(self.mass)
        y_mass = np.nansum(self.ypix.flatten() * self.mass) / np.nansum(self.mass)
        
        return x_mass, y_mass

    def update_orders(self, new_mmax, new_nmax, print_plots = True):
        
        """
        Update the Laguerre expansion orders (mmax and nmax) and recompute Laguerre coefficients.

        Parameters:
        -----------
        new_mmax : int
            New maximum order of the Laguerre expansion.
        new_nmax : int
            New maximum radial quantum number of the Laguerre expansion.

        Notes:
        ------
        This method updates the Laguerre expansion orders, recalculates Laguerre coefficients,
        and applies a mask to exclude certain pixels based on specific criteria.
        """
        
        self.mmax, self.nmax = new_mmax, new_nmax
        self.laguerre_amplitudes()

        
        # Apply mask as before
        gvals = np.where((self.R > self.galaxy_radius) | (self.mass < 0))
        self.R[gvals], self.phi[gvals], self.mass[gvals] = np.nan, np.nan, np.nan
        if print_plots == True:
            # Visualise the effect of the mask applied
            image_array = self.get_image_data(self.filename)
            plt.figure(figsize=(8, 6))
            plt.imshow(self.mass.reshape(image_array.shape).T, cmap='viridis') #undoing transpose
            plt.colorbar(label='Mask values')
            plt.title('Visualization of Mask')
            plt.xlabel('X Pixels')
            plt.ylabel('Y Pixels')
            plt.show()

    def laguerre_reconstruction(self, rr, pp):
        """
        Reconstruct the original image using Laguerre coefficients.

        Parameters:
        -----------
        rr : array-like, shape [x, y]
            Radial coordinates for reconstruction.
        pp : array-like, shape [x, y]
            Angular coordinates for reconstruction.

        Returns:
        --------
        reconstruction : ndarray
            Reconstructed image using Laguerre coefficients.

        Notes:
        ------
        This method reconstructs the original image using the Laguerre coefficients
        calculated from the Laguerre expansion.
        """
        nmvals = self._n_m()
        G_j = self._G_n(rr.flatten(), np.arange(0, self.nmax), self.rscl)
        G_j = G_j.reshape(G_j.shape[0], rr.shape[0], rr.shape[1]) #check this?
        fftotal = sum(
            self.coscoefs[m, n] * nmvals[m] * np.cos(m * pp) * G_j[n]
            + self.sincoefs[m, n] * nmvals[m] * np.sin(m * pp) * G_j[n]
            for m in range(self.mmax) for n in range(self.nmax)
        )

        return 0.5 * fftotal

    def compute_ratio(self, rscl, mmax, nmax, rval, phi, snapshotflat):
        """
        Compute a ratio metric using Laguerre amplitudes.

        Parameters:
        -----------
        rscl : float
            Scaling parameter for Laguerre functions.
        mmax : int
            Maximum angular order of the fourier expansion.
        nmax : int
            Maximum radial number of the Laguerre expansion.
        rval : float
            Radial coordinate for the Laguerre expansion.
        phi : float
            Angular coordinate for the Laguerre expansion.
        snapshotflat : array-like
            Flattened snapshot image data.

        Returns:
        --------
        ratio : float
            Ratio metric computed using Laguerre amplitudes.
        """
        
        self.rscl = rscl
        self.mmax = mmax
        self.nmax = nmax
        self.R = rval
        self.phi = phi
        self.mass = snapshotflat
        self.laguerre_amplitudes()
        abs_coscoefs_0 = abs(self.coscoefs[0, 0])
        norm_coscoefs_1 = np.linalg.norm(self.coscoefs[0, 1:])
        return norm_coscoefs_1 / abs_coscoefs_0
    


def BeefIt(filename, rscl_initial, mmax_initial, nmax_initial, new_mmax, new_nmax,
           print_plots = True, return_expander = False):
    """
    Process a galaxy by extracting data, computing Laguerre amplitudes, and generating plots.

    Parameters:
    galaxy_id (int): The ID of the galaxy to be processed.
    fits_files (list of str): List of paths to the FITS files.
    filename (str): Path to the mass catalog CSV file.

    Returns:
    - Centre and scale length details
    - Coefficient Chart
    - cosine coefficients, sine coefficients
    If print_plots = True, also returns:
        - Mask Image
        - Coefficient Chart
        - Coefficients, centre and scale length saved on HDF5 file
        - Display 4 contour maps including uncertainty maps
    """
    #read in image data, start expansion
    print('opening image')
    expander = ExpandGalaxies(filename, rscl_initial, mmax_initial, nmax_initial)
    av_image = expander.get_image_data()

    # Try and except block done here to bypass galaxy ID's which have empty image arrays. 
    try:
        print('getting pixel info, galaxy radius, mask, etc')
        rr, pp, xpix, ypix, fixed_image, xdim, ydim = expander.readsnapshot()
    
    except KeyError as e:
        print(f"Error processing galaxy")
        
        # Skip to the next galaxy ID
        return  

    # Calculate the Laguerre amplitudes
    expander.laguerre_amplitudes()

    # Find center using moments and optimization method
    print('finding center of image')
    x_mass, y_mass = expander.find_center_moments()
    print('center of mass estimate', x_mass, y_mass)
    center_x, center_y, _ = expander.find_center(initial_guess=(x_mass, y_mass)) #updates self.R, self.Phi for future steps
    print(f"Center of mass at x = {center_x}, y = {center_y}")
    
    # Test a range of rscl values to find the best one
    print('finding best rscl')
    min_ratio = 99999999#float('inf')
    best_rscl = None
    rscl_values = np.arange(1, min(xdim, ydim))
    for rscl in rscl_values:
        ratio = expander.compute_ratio(rscl, mmax_initial, nmax_initial, expander.R, expander.phi, fixed_image.flatten())
        #print('ratio shape', ratio.shape, ratio)
        #min_ratio_index = np.argmin(np.abs(ratio))
        #print('min_ratio_index', min_ratio_index)
        if np.abs(ratio) < np.abs(min_ratio):
            min_ratio = ratio
            best_rscl = rscl

    # Set the best rscl value
    expander.rscl = best_rscl

    print(f"New rscl value: {best_rscl}")

    # Update orders and recalculate the Laguerre amplitudes
    print('updating to new_mmax, new_nmax')
    expander.update_orders(new_mmax, new_nmax)
    
    expander.laguerre_amplitudes()




    if print_plots == True:
        image = np.asarray(Image.open(filename))
        av_image = np.mean(image, axis=2).T
        
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,10))
        ax1.imshow(fixed_image.T)
        ax1.set_title('masked image')
        ax2.imshow(av_image.T) 
        ax2.set_title('original image')
        # Perform the reconstruction with recentered rr, pp from pixel coords
        rr, pp = np.sqrt((xpix - center_x)**2 + (ypix - center_y)**2), np.arctan2(ypix - center_y, xpix - center_x)
        reconstruction = expander.laguerre_reconstruction(rr, pp)
        # Calculate the amplitude of the coefficients and create the plot using imshow
        amplitudes = np.sqrt(expander.coscoefs**2 + expander.sincoefs**2).T
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        c = ax.imshow(amplitudes, cmap='Blues', norm=LogNorm(), aspect='auto')
        fig.colorbar(c, ax=ax)
        ax.set_xlabel('Harmonics (m)')
        ax.set_ylabel('Radial Nodes (n)')
        ax.set_title('Amplitude of Fourier-Laguerre Coefficients')
        ax.set_xticks(np.arange(expander.mmax))
        ax.set_yticks(np.arange(expander.nmax))
        plt.tight_layout()
        plt.savefig(f'coeff.png', dpi=600)
        plt.show()

        # Create 4 subplots for visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
        cval = np.linspace(-5., 1., 32)

        # Plot in the following order: Original Image, Expanded Image, Relative Uncertainty and Absolute Uncertainty
        ax1.imshow((fixed_image).T)
        ax2.imshow((reconstruction).T)
        ax3.imshow(((reconstruction - fixed_image) / fixed_image).T,cmap='bwr',vmin=-1,vmax=1)
        ax4.imshow(abs(reconstruction - fixed_image).T,cmap='bwr',vmin=0,vmax=10)

        for ax, title in zip([ax1, ax2, ax3, ax4], ['log surface density', 'expanded surface density', 'relative uncertainty', 'absolute uncertainty']):
            ax.set_title(title)
            #ax.axis([-xdim/2, xdim/2, -xdim/2, xdim/2])
            ax.set_xticklabels(())
            ax.set_yticklabels(())
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(axis="y", which='both', direction="in")
            ax.tick_params(axis="x", which='both', direction="in", pad=5)
            #ax.scatter(0.0 if ax != ax1 else center_x, 0.0 if ax != ax1 else center_y, color='red', marker='x', label='(0,0)' if ax != ax1 else 'True Center')

        plt.tight_layout()
        plt.savefig(f'expand.png', dpi=600)
        plt.show()
    if return_expander == True:
        print('returning cosine coefficients, sine coefficients, and final expander class')
        return expander.coscoefs, expander.sincoefs, expander
    else:
        print('returning cosine coefficients, sine coefficients')
        return expander.coscoefs, expander.sincoefs   

