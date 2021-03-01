import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import helper
import mne
from mne.channels.layout import _find_topomap_coords
from mne.viz.topomap import _make_head_outlines
import pyvista as pv
import fooof
import ssd


def make_topoplot(
    values,
    info,
    ax,
    vmin=None,
    vmax=None,
    plot_head=True,
    cmap="RdBu_r",
    size=30,
    hemisphere="all",
    picks=None,
    pick_color=["#2d004f", "#254f00", "#000000"],
):
    """Makes an ECoG topo plot with electrodes circles, without interpolation.
    Modified from MNE plot_topomap to plot electrodes without interpolation.

    Parameters
    ----------
    values : array, 1-D
        Values to plot as color-coded circles.
    info : instance of Info
        The x/y-coordinates of the electrodes will be infered from this object.
    ax : instance of Axes
        The axes to plot to.
    vmin : float | None
         Lower bound of the color range. If None: - maximum absolute value.
    vmax : float | None
        Upper bounds of the color range. If None: maximum absolute value.
    plot_head : True | False
        Whether to plot the outline for the head.
    cmap : matplotlib colormap | None
        Colormap to use for values, if None, defaults to RdBu_r.
    size : int
        Size of electrode circles.
    picks : list | None
        Which electrodes should be highlighted with by drawing a thicker edge.
    pick_color : list
        Edgecolor for highlighted electrodes.
    hemisphere : string ("left", "right", "all")
        Restrict which hemisphere of head outlines coordinates to plot.

    Returns
    -------
    sc : matplotlib PathCollection
        The colored electrode circles.
    """

    pos = _find_topomap_coords(info, picks=None)
    sphere = np.array([0.0, 0.0, 0.0, 0.095])
    outlines = _make_head_outlines(
        sphere=sphere, pos=pos, outlines="head", clip_origin=(0.0, 0.0)
    )

    if plot_head:
        outlines_ = {
            k: v for k, v in outlines.items() if k not in ["patch", "mask_pos"]
        }
        for key, (x_coord, y_coord) in outlines_.items():
            if hemisphere == "left":
                if type(x_coord) == np.ndarray:
                    idx = x_coord <= 0
                    x_coord = x_coord[idx]
                    y_coord = y_coord[idx]
                ax.plot(x_coord, y_coord, color="k", linewidth=1, clip_on=False)
            elif hemisphere == "right":
                if type(x_coord) == np.ndarray:
                    idx = x_coord >= 0
                    x_coord = x_coord[idx]
                    y_coord = y_coord[idx]
                ax.plot(x_coord, y_coord, color="k", linewidth=1, clip_on=False)
            else:
                ax.plot(x_coord, y_coord, color="k", linewidth=1, clip_on=False)

    if not (vmin) and not (vmax):
        vmin = -values.max()
        vmax = values.max()

    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=size,
        edgecolors="grey",
        c=values,
        vmin=vmin,
        vmax=vmax,
        cmap=plt.get_cmap(cmap),
    )

    if np.any(picks):
        picks = np.array(picks)
        if picks.ndim > 0:
            if len(pick_color) == 1:
                pick_color = [pick_color] * len(picks)
            for i, idxx in enumerate(picks):
                ax.scatter(
                    pos[idxx, 0],
                    pos[idxx, 1],
                    s=size,
                    edgecolors=pick_color[i],
                    facecolors="None",
                    linewidths=1.5,
                    c=None,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=plt.get_cmap(cmap),
                )

        if picks.ndim == 2:
            if len(pick_color) == 1:
                pick_color = [pick_color] * len(picks)
            for i, idxx in enumerate(picks):
                ax.plot(
                    pos[idxx, 0],
                    pos[idxx, 1],
                    linestyle="-",
                    color=pick_color[i],
                    linewidth=1.5,
                )

    ax.axis("square")
    ax.axis("off")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    return sc


def reject_channels(raw, participant, experiment):
    """Exclude electrodes for resting datasets according to csv-file.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance from where to remove channels.
    participant : string
        Participant ID.
    experiment : string
        Experiment ID, for instance "fixation_pwrlaw".

    Returns
    -------
    raw : instance of Raw
        Raw instance with channels removed.

    """

    df = pd.read_csv("../csv/selected_datasets.csv")
    df = df[df.participant == participant]
    df = df[df.experiment == experiment]
    drop_chan = df.electrodes.values[0]
    idx_chan = ast.literal_eval(drop_chan)
    if len(idx_chan) > 0:
        ch_names = np.array(raw.ch_names)[idx_chan]
        raw.drop_channels(ch_names)

    return raw


def get_participant_peaks(participant, experiment, return_width=False):
    """Extract associated spectral peaks for a specific dataset.

    Parameters
    ----------
    participant : string
        Participant ID.
    experiment : string
        Experiment ID, for instance "fixation_pwrlaw".
    Returns
    -------
    peaks : array, 1-D
        The spectral peaks for a specific participant.
    """

    df_ssd = pd.read_csv("../csv/retained_components.csv")
    df_ssd = df_ssd[df_ssd.participant == participant]
    df_ssd = df_ssd[df_ssd.experiment == experiment]
    peaks = df_ssd.frequency.to_numpy()
    bandwidth = df_ssd.bandwidth.to_numpy()

    if return_width:
        return peaks, bandwidth
    else:
        return peaks


def compute_spatial_max(patterns, raw, plot=False):
    """Extracts the spatial maximum by interpolation.

    Parameters
    ----------
    patterns :
    raw : instance of Raw
        Raw instance for extracting electrode positions
    plot : bool
        If True, plot the spatial patterns with maxima marked for verification.

    Returns
    -------
    locations: array, 2-D
        The location of the spatial maximum for each pattern.
    """
    electrodes = np.array([r["loc"][:3] for r in raw.info["chs"]])

    hemisphere = helper.hemisphere(raw)
    if hemisphere == "left":
        electrodes[:, 0] = -electrodes[:, 0]

    x, y, z = electrodes[:, 0], electrodes[:, 1], electrodes[:, 2]
    ymin, ymax = min(y) - 15, max(y) + 15
    zmin, zmax = min(z) - 15, max(z) + 15
    yy, zz = np.mgrid[ymin:ymax:500j, zmin:zmax:500j]

    nr_patterns = patterns.shape[1]
    locations = np.zeros((nr_patterns, 3))
    interp_all = np.zeros((nr_patterns, 500, 500))
    for i in range(nr_patterns):
        idx_max = np.argmax(np.abs(patterns[:, i]))
        values = np.sign(patterns[idx_max, i]) * patterns[:, i]
        interp = scipy.interpolate.griddata(
            electrodes[:, 1:], values, xi=(yy, zz), method="cubic"
        )
        interp_all[i] = interp
        interp[np.isnan(interp)] = -np.Inf
        a = np.argmin(np.abs(interp - np.max(interp)))
        coords = np.unravel_index(a, interp.shape)

        max_y = yy[coords[0], coords[1]]
        max_z = zz[coords[0], coords[1]]

        distance = np.sum(
            np.abs(electrodes[:, 1:] - np.array([max_y, max_z])), axis=1
        )
        idx = np.argsort(distance)
        max_x = np.mean(electrodes[idx[:3], 0])
        locations[i] = [max_x, max_y, max_z]

    if plot:
        fig, ax = plt.subplots(1, nr_patterns)
        for i in range(nr_patterns):
            ax1 = ax[i]
            ax1.imshow(
                interp_all[i].T,
                origin="lower",
                cmap=plt.cm.RdBu_r,
                extent=[ymin, ymax, zmin, zmax],
                aspect="auto",
            )
            values = np.sign(patterns[idx_max, i]) * patterns[:, i]
            ax1.scatter(y, z, 10, c=values, marker=".")
            ax1.plot(locations[i, 1], locations[i, 2], "g.", markersize=10)

        fig.set_size_inches(15, 5)
        fig.show()

    return locations


def load_ssd(participant_id, peak):
    """Load spatial filters and patterns for a specific dataset.

    Parameters
    ----------
    participant : string
        Participant ID.
    peak : float
        frequency for which SSD filters and patterns should be loaded.
    Returns
    -------
    patterns : array, 2-D
        Spatial patterns as computed with the signal covariance matrix.
    filters : array, 2-D
        Spatial filters as computed by SSD.
    """

    file_name = "../results/ssd/ssd_%s_peak_%.2f.npy" % (
        participant_id,
        peak,
    )
    data = np.load(file_name, allow_pickle=True).item()
    filters = data["filters"]
    patterns = data["patterns"]

    return patterns, filters


def hemisphere(raw):
    """Return hemisphere of electrodes based on coordinates.

    Parameters
    ----------
    raw: instance of Raw

    Returns
    -------
    hemisphere: string
        Specifies on which side the electrode grid is placed.
    """
    xyz = np.array([ich["loc"][:3] for ich in raw.info["chs"]])

    if np.all(xyz[:, 0] < 0):
        hemisphere = "left"
    elif np.all(xyz[:, 0] < 0):
        hemisphere = "right"
    else:
        hemisphere = "both"

    return hemisphere


def get_camera_position():
    """ Camera position for pyvista 3d brains."""
    cpos = [
        (250.6143195554263, 41.91225218786929, 88.70813933287698),
        (31.183101868789514, -7.437521450446045, 14.637308132215361),
        (-0.3152066263035874, -0.022473087557700527, 0.948756946256487),
    ]

    return cpos


def plot_3d_brain(whichbrain="rightbrain"):
    """ Creates Pyvista object for one hemisphere."""

    data = scipy.io.loadmat("../data/motor_basic/halfbrains.mat")
    pos = data[whichbrain].item()[0]
    tri = data[whichbrain].item()[1] - 1
    tri = tri.astype("int")
    faces = np.hstack((3 * np.ones((tri.shape[0], 1)), tri))
    faces = faces.astype("int")
    brain_cloud = pv.PolyData(pos, faces)

    return brain_cloud


def plot_electrodes(raw):
    """ Create Pyvista object of the electrodes of 1 participant."""
    xyz = np.array([ich["loc"][:3] for ich in raw.info["chs"]])
    hemisphere = helper.hemisphere(raw)
    if hemisphere == "left":
        xyz[:, 0] = -xyz[:, 0]
    electrodes = pv.PolyData(xyz)

    return electrodes


def create_bipolar_derivation(raw, rows, prefix="bipolar"):
    """Convenience function for creating bipolar channels.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance containing traces for which bipolar derivation is taken.
    rows : list
        List of lists containing electrode indices, each list is treated
        separately. Bipolar channels are create between adjacent electrode
        entries.
    prefix : string
        Prefix for creating channel names.
    """
    counter = 0
    nr_channels = len(raw.ch_names)
    nr_bipolar_channels = len(rows) * (len(rows[0]) - 1)
    elecs = np.zeros((nr_bipolar_channels, 2), dtype="int")
    filters = np.zeros((nr_channels, nr_bipolar_channels))
    for row in rows:
        for i in range(len(row) - 1):
            filters[row[i], counter] = 1
            filters[row[i + 1], counter] = -1
            elecs[counter] = [row[i], row[i + 1]]
            counter += 1

    raw_bipolar = ssd.apply_filters(raw, filters, prefix=prefix)

    return raw_bipolar, filters, elecs


def plot_timeseries(
    ax, raw, cmap=["#2d004f", "#254f00", "#000000"], label=None
):
    """Convenience function for plotting raw traces.

    Parameters
    ----------
    ax : instance of Axes
        The axes to plot to.
    raw : instance of Raw
        Raw instance containing traces for which PSD is computed.
    cmap : list
        Colors for traces, specify for each channel.
    label : string
        Label to be plotted as title of axes.
    """

    nr_channels = len(raw.ch_names)
    for i_pick in range(nr_channels):
        signal = raw._data[i_pick]
        signal = signal / np.ptp(signal)
        ax.plot(raw.times, signal - i_pick, color=cmap[i_pick], lw=1)

    ymin = -nr_channels + 0.25
    ymax = 0.45
    ax.set(
        xlim=(raw.times[0], raw.times[-1]),
        xticks=np.arange(0, 2.01, 0.5),
        yticks=[],
        ylim=(ymin, ymax),
        title=label,
    )


def plot_psd(
    ax, raw, SNR, peak, cmap=["#2d004f", "#254f00", "#000000"], bin_width=0
):
    """Convenience function for plotting PSDs and SNR for a specific band.

    Parameters
    ----------
    ax : instance of Axes
        The axes to plot to.
    raw : instance of Raw
        Raw instance containing traces for which PSD is computed.
    SNR : array, 1-D
        array containing SNR values to be annotated, output of get_SNR.
    peak :
        Peak frequency to highlight.
    cmap :
        Colors for PSDs, specify for each channel.
    bin_width :
        Bandwidth to highlight, [peak-bin_width, peak-bin_width]
    """
    nr_seconds = 3

    n_fft = int(nr_seconds * raw.info["sfreq"])
    psd, freqs = mne.time_frequency.psd_welch(raw, fmin=1, fmax=70, n_fft=n_fft)

    for i in range(len(raw.ch_names)):
        ax.plot(freqs, 10 * np.log10(psd[i].T), color=cmap[i], lw=1)

    # plot aperiodic fit as example for first electrode
    fg = fooof.FOOOF()
    idx1 = 0
    fg.fit(freqs, psd[idx1])
    ax.plot(
        fg.freqs, 10 * fg._ap_fit, color=cmap[idx1], linestyle="--", alpha=0.8
    )
    idx_max = np.argmax(psd[idx1])
    peak_freq = freqs[idx_max]
    peak_freq = fooof.analysis.get_band_peak_fm(fg, [peak - 2, peak + 2])[0]

    ax.plot(
        [peak_freq, peak_freq],
        [10 * fg._ap_fit[idx_max], 10 * np.log10(psd[idx1][idx_max])],
        lw=1,
        zorder=-3,
        color=cmap[idx1],
    )

    for i in range(len(raw.ch_names)):
        ax.text(
            45,
            -4.5 * i + 0.95 * ax.get_ylim()[1],
            "SNR=%.1f dB" % SNR[i],
            color=cmap[i],
            ha="right",
        )

    # set axes properties
    ax.set(
        xlim=(1, 45),
        xlabel="frequency [Hz]",
        ylabel="log PSD",
        xticks=[10, 20, 30, 40],
        yticklabels=[],
    )

    # highlight frequency band
    if bin_width > 0:
        ax.axvspan(
            peak - bin_width,
            peak + bin_width,
            color="gray",
            alpha=0.25,
            zorder=-3,
        )


def get_SNR(raw, fmin=1, fmax=55, seconds=3, freq=[8, 13]):
    """Compute power spectrum and calculate 1/f-corrected SNR in one band.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance containing traces for which to compute SNR
    fmin : float
        minimum frequency that is used for fitting spectral model.
    fmax : float
        maximum frequency that is used for fitting spectral model.
    seconds: float
        Window length in seconds, converts to FFT points for PSD calculation.
    freq : list | [8, 13]
        SNR in that frequency window is computed.
    Returns
    -------
    SNR : array, 1-D
        Contains SNR (1/f-corrected, for a chosen frequency) for each channel.
    """
    SNR = np.zeros((len(raw.ch_names),))
    n_fft = int(seconds * raw.info["sfreq"])
    psd, freqs = mne.time_frequency.psd_welch(
        raw, fmin=fmin, fmax=fmax, n_fft=n_fft
    )

    fm = fooof.FOOOFGroup()
    fm.fit(freqs, psd)

    for pick in range(len(raw.ch_names)):
        psd_corr = 10 * np.log10(psd[pick]) - 10 * fm.get_fooof(pick)._ap_fit
        idx = np.where((freqs > freq[0]) & (freqs < freq[1]))[0]
        idx_max = np.argmax(psd_corr[idx])
        SNR[pick] = psd_corr[idx][idx_max]

    return SNR
