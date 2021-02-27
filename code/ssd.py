""" Functions to compute Spatial-Spectral Decompostion (SSD).

Reference
---------
Nikulin VV, Nolte G, Curio G.: A novel method for reliable and fast
extraction of neuronal EEG/MEG oscillations on the basis of
spatio-spectral decomposition. Neuroimage. 2011 Apr 15;55(4):1528-35.
doi: 10.1016/j.neuroimage.2011.01.057. Epub 2011 Jan 27. PMID: 21276858.

"""

import numpy as np
from scipy.linalg import eig
import mne


def compute_ged(cov_signal, cov_noise):
    """Compute a generatlized eigenvalue decomposition maximizing principal
    directions spanned by the signal contribution while minimizing directions
    spanned by the noise contribution.

    Parameters
    ----------
    cov_signal : array, 2-D
        Covariance matrix of the signal contribution.
    cov_noise : array, 2-D
        Covariance matrix of the noise contribution.

    Returns
    -------
    filters : array
        SSD spatial filter matrix, columns are individual filters.

    """

    nr_channels = cov_signal.shape[0]

    # check for rank-deficiency
    [lambda_val, filters] = eig(cov_signal)
    idx = np.argsort(lambda_val)[::-1]
    filters = np.real(filters[:, idx])
    lambda_val = np.real(lambda_val[idx])
    tol = lambda_val[0] * 1e-6
    r = np.sum(lambda_val > tol)

    # if rank smaller than nr_channels make expansion
    if r < nr_channels:
        print("Warning: Input data is not full rank")
        M = np.matmul(filters[:, :r], np.diag(lambda_val[:r] ** -0.5))
    else:
        M = np.diag(np.ones((nr_channels,)))

    cov_signal_ex = (M.T @ cov_signal) @ M
    cov_noise_ex = (M.T @ cov_noise) @ M

    [lambda_val, filters] = eig(cov_signal_ex, cov_signal_ex + cov_noise_ex)

    # eigenvalues should be sorted by size already, but double checking
    idx = np.argsort(lambda_val)[::-1]
    filters = filters[:, idx]
    filters = np.matmul(M, filters)

    return filters


def apply_filters(raw, filters, prefix="ssd"):
    """Apply spatial filters on continuous data.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance with signals to be spatially filtered.
    filters : array, 2-D
        Spatial filters as computed by SSD.
    prefix : string | None
        Prefix for renaming channels for disambiguation. If None: "ssd"
        is used.

    Returns
    -------
    raw_projected : instance of Raw
        Raw instance with projected signals as traces.
    """

    raw_projected = raw.copy()
    components = filters.T @ raw.get_data()
    nr_components = filters.shape[1]
    raw_projected._data = components

    ssd_channels = ["%s%i" % (prefix, i + 1) for i in range(nr_components)]
    mapping = dict(zip(raw.info["ch_names"], ssd_channels))
    mne.channels.rename_channels(raw_projected.info, mapping)
    raw_projected.drop_channels(raw_projected.info["ch_names"][nr_components:])

    return raw_projected


def compute_patterns(cov_signal, filters):
    """Compute spatial patterns for a specific covariance matrix.

    Parameters
    ----------
    cov_signal : array, 2-D
        Covariance matrix of the signal contribution.
    filters : array, 2-D
        Spatial filters as computed by SSD.
    Returns
    -------
    patterns : array, 2-D
        Spatial patterns.
    """

    top = cov_signal @ filters
    bottom = (filters.T @ cov_signal) @ filters
    patterns = top @ np.linalg.pinv(bottom)

    return patterns


def run_ssd(raw, peak, band_width):
    """Wrapper for compute_ssd with standard settings for definining filters.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance with signals to be spatially filtered.
    peak : float
        Peak frequency of the desired signal contribution.
    band_width : float
        Spectral bandwidth for the desired signal contribution.

    Returns
    -------
    filters : array, 2-D
        Spatial filters as computed by SSD, each column = 1 spatial filter.
    patterns : array, 2-D
        Spatial patterns, with each pattern being a column vector.
    """

    signal_bp = [peak - band_width, peak + band_width]
    noise_bp = [peak - (band_width + 2), peak + (band_width + 2)]
    noise_bs = [peak - (band_width + 1), peak + (band_width + 1)]

    filters, patterns = compute_ssd(raw, signal_bp, noise_bp, noise_bs)

    return filters, patterns


def compute_ssd(raw, signal_bp, noise_bp, noise_bs):
    """Compute SSD for a specific peak frequency.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance with signals to be spatially filtered.
    signal_bp : tuple
        Pass-band for defining the signal contribution. E.g. (8, 13)
    noise_bp : tuple
        Pass-band for defining the noise contribution.
    noise_bs : tuple
        Stop-band for defining the noise contribution.


    Returns
    -------
    filters : array, 2-D
        Spatial filters as computed by SSD, each column = 1 spatial filter.
    patterns : array, 2-D
        Spatial patterns, with each pattern being a column vector.
    """

    iir_params = dict(order=2, ftype="butter", output="sos")

    # bandpass filter for signal
    raw_signal = raw.copy().filter(
        l_freq=signal_bp[0],
        h_freq=signal_bp[1],
        method="iir",
        iir_params=iir_params,
        verbose=False,
    )

    # bandpass filter
    raw_noise = raw.copy().filter(
        l_freq=noise_bp[0],
        h_freq=noise_bp[1],
        method="iir",
        iir_params=iir_params,
        verbose=False,
    )

    # bandstop filter
    raw_noise = raw_noise.filter(
        l_freq=noise_bs[1],
        h_freq=noise_bs[0],
        method="iir",
        iir_params=iir_params,
        verbose=False,
    )

    # compute covariance matrices for signal and noise contributions

    if raw_signal._data.ndim == 3:
        cov_signal = mne.compute_covariance(raw_signal, verbose=False).data
        cov_noise = mne.compute_covariance(raw_noise, verbose=False).data
    elif raw_signal._data.ndim == 2:
        cov_signal = np.cov(raw_signal._data)
        cov_noise = np.cov(raw_noise._data)

    # compute spatial filters
    filters = compute_ged(cov_signal, cov_noise)

    # compute spatial patterns
    patterns = compute_patterns(cov_signal, filters)

    return filters, patterns
