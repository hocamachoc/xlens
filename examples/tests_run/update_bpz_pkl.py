import os
import glob
import numpy as np


NUM_Z_GRIDS = 401
Z_MAX = 4.0
Z_GRIDS = np.linspace(0.0, Z_MAX, NUM_Z_GRIDS)
FILTER_LIST = ["DC2LSST_g", "DC2LSST_r", "DC2LSST_i", "DC2LSST_z", "DC2LSST_y"]
SPECTRA_FILE = "cosmossedswdust136.list"
MAG_ERR_MIN = 0.005
MAG_ERR_MAX = 20.0


def _load_templates(data_path):
    from desc_bpz.useful_py3 import get_str, get_data, match_resol
    # The redshift range we will evaluate on
    z = Z_GRIDS
    filters = FILTER_LIST
    spectra_file = os.path.join(data_path, "SED", SPECTRA_FILE)
    spectra = [s[:-4] for s in get_str(spectra_file)]
    nt = len(spectra)
    nf = len(filters)
    nz = len(z)
    flux_templates = np.zeros((nz, nt, nf))
    ab_dir = os.path.join(data_path, "AB")
    # make a list of all available AB files in the AB directory
    ab_file_list = glob.glob(ab_dir + "/*.AB")
    ab_file_db = [os.path.split(x)[-1] for x in ab_file_list]
    for i, s in enumerate(spectra):
        for j, f in enumerate(filters):
            model = f"{s}.{f}.AB"
            model_path = os.path.join(data_path, "AB", model)
            zo, f_mod_0 = get_data(model_path, (0, 1))
            flux_templates[:, i, j] = match_resol(zo, f_mod_0, z)
    return flux_templates

def _preprocess_magnitudes(self, data):
    from desc_bpz.bpz_tools_py3 import e_mag2frac

    bands = self.config.bands
    errs = self.config.err_bands

    fluxdict = {}

    # Load the magnitudes
    zp_frac = e_mag2frac(np.array(self.config.zp_errors))
    # Group the magnitudes and errors into one big array
    mags = np.array([data[b] for b in bands]).T
    mag_errs = np.array([data[er] for er in errs]).T
    np.clip(mag_errs, MAG_ERR_MIN, MAG_ERR_MAX, mag_errs)
    # Convert to pseudo-fluxes
    flux = 10.0**(-0.4 * mags)
    flux_err = flux * (10.0**(0.4 * mag_errs) - 1.0)
    add_err = ((zp_frac * flux)**2)
    flux_err = np.sqrt(flux_err**2 + add_err)
    # Upate the flux dictionary with new things we have calculated
    fluxdict['flux'] = flux
    fluxdict['flux_err'] = flux_err
    m_0_col = self.config.bands.index(self.config.ref_band)
    fluxdict['mag0'] = mags[:, m_0_col]
    return fluxdict

def predict(flux_templates, flux, flux_err, mag_0, model_dict=None):
    from desc_bpz.bpz_tools_py3 import p_c_z_t
    from desc_bpz.prior_from_dict import prior_function

    nt = flux_templates.shape[1]

    # The likelihood and prior...
    pczt = p_c_z_t(flux, flux_err, flux_templates)
    L = pczt.likelihood

    # old prior code returns NoneType for prior if "flat" or "none"
    # just hard code the no prior case for now for backward compatibility
    if model_dict is None:
        P = np.ones(L.shape)
    else:
        # set num templates to nt, which is hardcoding to "interp=0"
        # in BPZ, i.e. do not create any interpolated templates
        P = prior_function(z, mag_0, model_dict, nt)

    post = L * P
    # Right now we jave the joint PDF of p(z,template). Marginalize
    # over the templates to just get p(z)
    post_z = post.sum(axis=1)
    zpos = np.argmax(post_z)
    zmode = Z_GRIDS[zpos]
    return zmode

