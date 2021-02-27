""" This figure shows examples where SSD improves SNR of oscillations in sEEG.
"""
import mne
import numpy as np
import matplotlib.pyplot as plt
import helper
import ssd
import matplotlib.gridspec as gridspec
import fooof
import pyvista as pv
import matplotlib.image as mpimg


def plot_seeg_3dbrain(raw):
    """Convenience function for plotting 3d brain and the 3 leads."""

    # camera position for ventral view
    cpos = [
        (192.1355480872916, -142.5172055994041, -261.47527261474386),
        (24.065850569725583, -19.416839467317363, 15.642167878335641),
        (0.20039110730958792, 0.9346693866417147, -0.29366058943283213),
    ]

    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=[350, 700])
    cloud = helper.plot_3d_brain()
    actor = plotter.add_mesh(cloud, opacity=1)
    xyz_all = np.array([ich["loc"][:3] for ich in raw.info["chs"]])
    picks_all = [range(8), range(8, 14), range(14, 19)]
    colors = ["purple", "#41A859", "dodgerblue"]

    # plot each lead in a different color
    for i, picks in enumerate(picks_all):
        ch_names = ["ecog%i" % i for i in picks]
        lead1 = raw.copy().pick_channels(ch_names)
        xyz = np.array([ich["loc"][:3] for ich in lead1.info["chs"]])

        # slightly adjust coordinates to prevent electrodes from being invisible
        xyz[:, 0] += 4
        xyz[:, 2] -= 4

        electrodes = pv.PolyData(xyz)
        act_electrodes = plotter.add_mesh(
            electrodes,
            point_size=25,
            color=colors[i],
            render_points_as_spheres=True,
        )

    xyz_all[:, 0] += 4
    xyz_all[:, 2] -= 4
    for ii, pick in enumerate(picks1):

        electrodes = pv.PolyData(xyz_all[pick])
        act_electrodes = plotter.add_mesh(
            electrodes,
            point_size=40,
            color=colors_sel[ii],
            render_points_as_spheres=True,
        )

    plotter.show(
        cpos=cpos, title=str(cpos), screenshot="../figures/3dbrain_seeg.png"
    )
    plotter.screenshot(
        "../figures/3dbrain_seeg.png", transparent_background=True
    )
    plotter.close()


def plot_SNR(freq, psd, ax, color, x=5, y=51, peak=9.5, plot=True):
    """ Convenience function for extracting and plotting SNR. """
    fg = fooof.FOOOF()
    fg.fit(freq, psd)
    peak_freq = fooof.analysis.get_band_peak_fm(fg, [peak - 1, peak + 1])[0]
    idx_max = np.argmin(np.abs(freq - peak_freq))
    if plot:
        ax.plot(
            fg.freqs, 10 * fg._ap_fit, color=color, linestyle="--", alpha=0.8
        )
        ax.plot(
            [peak_freq, peak_freq],
            [10 * fg._ap_fit[idx_max], 10 * np.log10(psd[idx_max])],
            lw=1,
            zorder=-3,
            color=color,
        )
    SNR = 10 * np.log10(psd[idx_max]) - 10 * fg._ap_fit[idx_max]
    ax.text(x, y, "SNR=%.1f dB" % SNR, color=color)


# -- set colors & PSD parameters
plt.ion()
nr_seconds = 3
fmin = 2
fmax = 55

colors_bi = ["#000000", "#444444", "#888888"]
colors_sel = ["purple", "#41A859", "dodgerblue"]
colors_ssd = ["#CC2B1C", "#3C2C71"]

# -- load continuous data
participant_id = "ja_faces_basic"
raw_file = "../working/%s_raw.fif" % participant_id
raw = mne.io.read_raw_fif(raw_file)
raw.load_data()
raw.pick_types(ecog=True)

# -- select 3 sEEG leads
picks = range(19)
ch_names = ["ecog%i" % i for i in picks]
raw.pick_channels(ch_names)
raw2 = raw.copy()

ch_names_leads = ["ecog3", "ecog13", "ecog18"]
picks1 = mne.pick_channels(raw2.ch_names, ch_names_leads, ordered=True)
plot_seeg_3dbrain(raw)

# -- apply SSD
peak1 = 4.0
bin_width1 = 1.25
filters1, patterns = ssd.run_ssd(raw, peak1, bin_width1)

peak2 = 8.15
bin_width2 = 2.5
filters2, patterns2 = ssd.run_ssd(raw, peak2, bin_width2)

# combine top-SNR filters for each peak frequency into one matrix
filters1[:, 1] = filters2[:, 0]
patterns[:, 1] = patterns2[:, 0]
raw_ssd = ssd.apply_filters(raw, filters1[:, :2])
raw_ssd.filter(1, None)

picks = range(8)
ch_names = ["ecog%i" % i for i in picks]
lead1 = raw.copy().pick_channels(ch_names)
lead1.set_eeg_reference("average")


picks = range(8, 14)
ch_names = ["ecog%i" % i for i in picks]
lead2 = raw.copy().pick_channels(ch_names)
lead2.set_eeg_reference("average")

picks = range(14, 19)
ch_names = ["ecog%i" % i for i in picks]
lead3 = raw.copy().pick_channels(ch_names)
lead3.set_eeg_reference("average")


lead1.add_channels([lead2, lead3])
raw = lead1
raw.filter(1, None)


# -- apply bipolar filtering
filters = np.zeros((len(raw.ch_names), 2))
filters[7, 0] = 1
filters[8, 0] = -1
filters[17, 1] = 1
filters[18, 1] = -1
raw_bipolar = ssd.apply_filters(raw2, filters, prefix="bipolar")


# -- compute PSD for CAR, bipolar & SSD signals
raw.pick_channels(ch_names_leads)
n_fft = int(nr_seconds * raw.info["sfreq"])

psd, freq = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, n_fft=n_fft)

psd_ssd, freq = mne.time_frequency.psd_welch(
    raw_ssd, fmin=fmin, fmax=fmax, n_fft=n_fft
)

psd_bipolar, freq = mne.time_frequency.psd_welch(
    raw_bipolar, fmin=fmin, fmax=fmax, n_fft=n_fft
)

# -- crop a time interval with some oscillations
tmin = 143
tmax = tmin + 2.5
raw.crop(tmin, tmax)
raw_ssd.crop(tmin, tmax)
raw_bipolar.crop(tmin, tmax)

# -- create plot
fig, ax = plt.subplots(figsize=(7.5, 5))
gs = gridspec.GridSpec(3, 4)

# -- plot brain
ax = plt.subplot(gs[:2, 0])

img = mpimg.imread("../figures/3dbrain_seeg.png")
ax.imshow(img)
ax.axis("off")


# collect all information in an unwieldy way
cond = (
    [
        raw,
        psd,
        colors_sel,
        [True, True, False],
        ["CAR\ne1", "CAR\ne2", "CAR\ne3"],
        [peak1, peak2, peak1],
    ],
    [
        raw_bipolar,
        psd_bipolar,
        colors_bi,
        [True, True],
        ["bipolar\ne1", "bipolar\ne2"],
        [peak2, peak1, peak1],
    ],
    [
        raw_ssd,
        psd_ssd,
        colors_ssd,
        [True, True],
        [
            "SSD component\npeak frequency= %.1f Hz" % peak1,
            "SSD component\npeak frequency= %.1f Hz" % peak2,
        ],
        [peak1, peak2],
    ],
)

# -- plot time domain signals
ax = plt.subplot(gs[:, 1:3])

counter = 0
for i_cond in range(3):
    raw1, psd1, colors1, _, labels, _ = cond[i_cond]
    for i in range(len(raw1.ch_names)):
        signal = raw1._data[i]
        signal = signal / np.ptp(signal)
        ax.plot(raw.times, signal - counter, color=colors1[i], lw=1)
        ax.text(
            -0.1,
            -counter - 0.1,
            labels[i],
            va="center",
            ha="right",
            color=colors1[i],
        )
        counter += 1
    counter += 0.5

ax.set(xlim=(raw.times[0], raw.times[-1]), xlabel="time [s]", yticks=[])

# -- plot spectrum for all referencing types
for i_cond in range(3):
    ax = plt.subplot(gs[i_cond, 3])
    raw1, psd1, colors1, plot1, _, peaks = cond[i_cond]
    for i in range(len(raw1.ch_names)):
        ax.plot(freq, 10 * np.log10(psd1[i]).T, color=colors1[i], lw=1)
        plot_SNR(
            freq,
            psd1[i],
            ax,
            colors1[i],
            peak=peaks[i],
            x=12,
            y=10 * np.log10(psd1[0, 0]) - i * 5,
            plot=plot1[i],
        )
    if i_cond == 2:
        ax.set(xlabel="frequency [Hz]")
    ax.axvspan(
        peak1 - bin_width1,
        peak1 + bin_width1,
        color=colors_ssd[0],
        alpha=0.25,
        zorder=-3,
    )

    ax.axvspan(
        peak2 - bin_width2,
        peak2 + bin_width2,
        color=colors_ssd[1],
        alpha=0.25,
        zorder=-3,
    )

fig.savefig("../figures/fig3_example_sEEG.pdf", dpi=300)
