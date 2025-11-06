import numpy as np
import scipy.interpolate
import pyccl as ccl

# New imports for multi-bin functionality
import itertools
from scipy.fft import fht, ifht, fhtoffset # Assuming these are available or implemented

class LogNormalShearFlat:
    def __init__(self, cosmo, dndz_list, z_bins, field_size_deg=5.0, npix=2048, seed=None):
        """
        Initializes the LogNormalShearFlat class for multi-bin log-normal shear.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl cosmology object.
        dndz_list : list of tuples
            A list where each element is a (z_arr, dndz_arr) tuple defining the
            redshift distribution for a single tomographic bin.
        z_bins : array_like
            The edges of the redshift bins. Used to determine which bin a galaxy belongs to.
        field_size_deg : float, optional
            The side length of the square simulation field in degrees. Defaults to 5.0.
        npix : int, optional
            The number of pixels per side for the internal simulation grid. Defaults to 2048.
        seed : int, optional
            A random seed for reproducibility. Defaults to None.
        """
        self.cosmo = cosmo
        self.dndz_list = dndz_list
        self.z_bins = np.asarray(z_bins)
        self.n_bins = len(self.dndz_list)
        self.field_size_deg = field_size_deg
        self.npix = npix
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Define ell range for power spectrum calculation
        field_size_rad = np.deg2rad(self.field_size_deg)
        pixel_scale_rad = field_size_rad / self.npix
        l_min = 2 * np.pi / field_size_rad
        l_max = np.pi / pixel_scale_rad
        ell = np.logspace(np.log10(l_min), np.log10(l_max), 1024)
        self.ell = ell # Store for later use in conversion

        # --- New: Generate C_ij^kappa(ell) from dndz_list ---
        self.c_ij_kappa = self._generate_power_spectra_from_dndz()

        # --- New: Convert C_ij^kappa(ell) to C_ij^mu(ell) ---
        self.c_ij_mu = self._power_spectra_conversion()

        # --- Existing: Create k-space grid (remains largely the same) ---
        k_modes = np.fft.fftfreq(self.npix, d=pixel_scale_rad) * 2 * np.pi
        kx, ky = np.meshgrid(k_modes, k_modes)
        k_abs = np.sqrt(kx**2 + ky**2)

        # Interpolate power spectrum to k_abs grid
        # Avoid division by zero at k_abs = 0
        k_abs[k_abs == 0] = 1e-10

        # --- New: Generate Correlated Gaussian Fields (Fourier Space) ---
        gaussian_fields_fourier = self._generate_correlated_gaussian_fields_fourier(
            k_abs, self.ell, self.c_ij_mu, field_size_rad, kx, ky
        )

        # --- New: Transform to Real Space (N kappa_G maps) ---
        self.kappa_G_maps = np.zeros((self.n_bins, self.npix, self.npix))
        for i in range(self.n_bins):
            self.kappa_G_maps[i, :, :] = np.fft.ifft2(gaussian_fields_fourier[i, :, :]).real

        # --- New: Apply Log-Normal Transform (N kappa_LN maps) ---
        self.kappa_LN_maps = np.zeros_like(self.kappa_G_maps)
        for i in range(self.n_bins):
            sigma_G_squared = np.var(self.kappa_G_maps[i, :, :])
            self.kappa_LN_maps[i, :, :] = np.exp(self.kappa_G_maps[i, :, :] - sigma_G_squared / 2) - 1

        # --- New: Generate Shear Fields (N gamma1, gamma2 maps in Fourier space) ---
        gamma1_fourier_maps = np.zeros_like(gaussian_fields_fourier, dtype=complex)
        gamma2_fourier_maps = np.zeros_like(gaussian_fields_fourier, dtype=complex)
        for i in range(self.n_bins):
            kappa_LN_fourier = np.fft.fft2(self.kappa_LN_maps[i, :, :])
            gamma1_fourier_maps[i, :, :] = ((kx**2 - ky**2) / k_abs**2) * kappa_LN_fourier
            gamma2_fourier_maps[i, :, :] = ((2 * kx * ky) / k_abs**2) * kappa_LN_fourier

        # --- New: Transform Shear to Real Space (N gamma1, gamma2 maps) ---
        self.gamma1_maps = np.zeros_like(self.kappa_G_maps)
        self.gamma2_maps = np.zeros_like(self.kappa_G_maps)
        for i in range(self.n_bins):
            self.gamma1_maps[i, :, :] = np.fft.ifft2(gamma1_fourier_maps[i, :, :]).real
            self.gamma2_maps[i, :, :] = np.fft.ifft2(gamma2_fourier_maps[i, :, :]).real

        # --- New: Create Interpolators (lists of interpolators) ---
        x_coords = np.linspace(-self.field_size_deg / 2, self.field_size_deg / 2, self.npix)
        y_coords = np.linspace(-self.field_size_deg / 2, self.field_size_deg / 2, self.npix)

        self.kappa_interps = []
        self.gamma1_interps = []
        self.gamma2_interps = []

        for i in range(self.n_bins):
            self.kappa_interps.append(scipy.interpolate.RectBivariateSpline(x_coords, y_coords, self.kappa_LN_maps[i, :, :]))
            self.gamma1_interps.append(scipy.interpolate.RectBivariateSpline(x_coords, y_coords, self.gamma1_maps[i, :, :]))
            self.gamma2_interps.append(scipy.interpolate.RectBivariateSpline(x_coords, y_coords, self.gamma2_maps[i, :, :]))

    def _generate_power_spectra_from_dndz(self):
        """
        Generates the C_ij^kappa(ell) power spectra matrix using pyccl
        from the provided dndz_list.
        """
        c_ij_kappa = np.zeros((self.n_bins, self.n_bins, len(self.ell)))
        tracers = [ccl.WeakLensingTracer(self.cosmo, dndz=dndz) for dndz in self.dndz_list]

        # Use itertools to compute the upper triangle of the power spectra matrix
        for i, j in itertools.combinations_with_replacement(range(self.n_bins), 2):
            cl = ccl.angular_cl(self.cosmo, tracers[i], tracers[j], self.ell)
            c_ij_kappa[i, j, :] = cl
            if i != j:
                c_ij_kappa[j, i, :] = cl # Mirror for lower triangle
        return c_ij_kappa

    def _power_spectra_conversion(self):
        """
        Converts the log-normal power spectra C_ij^kappa(ell) to the Gaussian
        power spectra C_ij^mu(ell) using Hankel transforms.
        """
        n_ell = len(self.ell)
        # Define theta range for Hankel transform.
        # This needs careful consideration for flat-sky vs curved-sky.
        # For flat-sky, ell is k, and theta is real-space separation.
        # The choice of theta_min/max and dln_theta is crucial.
        # A simple approach for now, similar to MultiBinLogNormalShear:
        theta_min = np.pi / np.max(self.ell)
        theta_max = np.pi / np.min(self.ell)
        dln_theta = np.log(theta_max / theta_min) / (n_ell - 1)
        # self.theta = np.exp(np.linspace(np.log(theta_min), np.log(theta_max), n_ell)) # Not strictly needed to store

        c_ij_mu = np.zeros_like(self.c_ij_kappa)
        for i, j in itertools.combinations_with_replacement(range(self.n_bins), 2):
            # Hankel transform C_kappa(ell) to xi_kappa(theta)
            # The fhtoffset parameter needs to be chosen carefully based on the definition
            # of the Hankel transform used. For C_ell to xi(theta), mu=0 is typical.
            offset = fhtoffset(dln_theta, mu=0)
            # Ensure input to ifht is 1D
            xi_ij_kappa = ifht(self.c_ij_kappa[i, j, :], dln_theta, mu=0, offset=offset)

            # Apply log-normal relation in real space
            xi_mu = np.log(1 + xi_ij_kappa)

            # Inverse Hankel transform xi_mu(theta) back to C_mu(ell)
            c_ij_mu[i, j, :] = fht(xi_mu, dln_theta, mu=0, offset=offset)

            if i != j:
                c_ij_mu[j, i, :] = c_ij_mu[i, j, :]
        return c_ij_mu

    def _generate_correlated_gaussian_fields_fourier(self, k_abs, ell_theory, c_ij_mu, field_size_rad, kx, ky):
        """
        Generates N correlated Gaussian fields in Fourier space.
        """
        # Interpolate C_ij^mu(ell) to the k_abs grid
        # c_ij_mu_interp will be (N, N, npix, npix)
        c_ij_mu_interp = np.zeros((self.n_bins, self.n_bins, self.npix, self.npix))
        for i, j in itertools.product(range(self.n_bins), repeat=2):
            c_ij_mu_interp[i, j, :, :] = np.interp(k_abs, ell_theory, c_ij_mu[i, j, :], left=0.0, right=0.0)

        # Generate N independent complex Gaussian random fields
        # The 1/sqrt(2) is for the complex random numbers, which have variance 1 in real and imag parts.
        z_fields = (
            np.random.normal(size=(self.n_bins, self.npix, self.npix)) +
            1j * np.random.normal(size=(self.n_bins, self.npix, self.npix))
        ) / np.sqrt(2.0)

        # Apply Cholesky decomposition for each k-mode
        # The result will be (N, npix, npix) for the N correlated fields
        correlated_gaussian_fields_fourier = np.zeros_like(z_fields)

        # Scale power spectrum for grid and generate field
        # Var(FFT(kappa)) = (N_pix^4 / L^2) * C_l, where L is field size in radians.
        # A factor of 2 is needed to compensate for power loss when taking .real of ifft of non-hermitian field.
        pk_scaling_factor = 2 * (self.npix**4 / (field_size_rad**2))

        for x, y in itertools.product(range(self.npix), repeat=2):
            # Get the C_mu matrix for this specific k-mode
            C_mu_k = c_ij_mu_interp[:, :, x, y]

            # Ensure C_mu_k is positive semi-definite (e.g., add a small diagonal term if needed)
            # This can be an issue with interpolation or very small values.
            # For now, assume it's well-behaved.
            try:
                L_k = np.linalg.cholesky(C_mu_k)
            except np.linalg.LinAlgError:
                # Handle cases where C_mu_k might not be positive definite
                # e.g., by adding a small diagonal term or setting L_k to identity
                # For now, let's add a small diagonal term for robustness.
                print(f"Warning: Cholesky decomposition failed at k-mode ({x}, {y}). "
                      "Matrix might not be positive definite. Adding a small diagonal term.")
                C_mu_k += np.eye(self.n_bins) * 1e-12 # Small regularization
                L_k = np.linalg.cholesky(C_mu_k)

            # Apply Cholesky decomposition to the independent Gaussian random numbers
            # L_k is (N, N), z_fields[:, x, y] is (N,)
            correlated_gaussian_fields_fourier[:, x, y] = np.dot(L_k, z_fields[:, x, y])

        # Apply the scaling factor to the generated fields
        # This scaling factor is applied to the variance, so sqrt(factor) to the field itself.
        correlated_gaussian_fields_fourier *= np.sqrt(pk_scaling_factor)

        return correlated_gaussian_fields_fourier

    def distort_galaxy(self, src):
        # Input dx, dy are in arcseconds, convert to degrees
        dx_deg = src['dx'] / 3600.0
        dy_deg = src['dy'] / 3600.0
        galaxy_z = src['z']

        # Determine the correct redshift bin
        # np.digitize returns the index of the bin to which each value in input array belongs.
        # The bins are defined by z_bins. If z_bins = [z0, z1, z2], then bin 0 is [z0, z1), bin 1 is [z1, z2).
        # We subtract 1 to get 0-indexed bin_index.
        bin_index = np.digitize(galaxy_z, self.z_bins) - 1

        # Handle galaxies outside the defined redshift bins
        if bin_index < 0 or bin_index >= self.n_bins:
            # Return zero shear/kappa (no lensing effect)
            return {'gamma1': 0.0, 'gamma2': 0.0, 'kappa': 0.0}

        # Use the interpolators for the selected bin
        kappa = self.kappa_interps[bin_index](dx_deg, dy_deg)[0, 0]
        gamma1 = self.gamma1_interps[bin_index](dx_deg, dy_deg)[0, 0]
        gamma2 = self.gamma2_interps[bin_index](dx_deg, dy_deg)[0, 0]

        return {'gamma1': gamma1, 'gamma2': gamma2, 'kappa': kappa}