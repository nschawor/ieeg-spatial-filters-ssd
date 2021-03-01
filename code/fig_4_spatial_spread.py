""" Figure shows example of two close-by rhythms and illustrates spatial spread.
"""
import mne
import numpy as np
import helper
import ssd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.spatial.distance import pdist, squareform

plt.ion()

# -- load continuous data
participant_id = "wc_fixation_pwrlaw"
raw = mne.io.read_raw_fif("../working/%s_raw.fif" % participant_id)
raw.load_data()
raw.pick_types(ecog=True)

# -- compute & apply SSD spatial filters
peak = 8.15
bin_width = 2
filters, patterns = ssd.run_ssd(raw, peak, bin_width)

nr_components = 2
raw_ssd = ssd.apply_filters(raw, filters[:, :nr_components])
picks = [np.argmax(np.abs(patterns[:, 0])), np.argmax(np.abs(patterns[:, 1]))]
raw2 = raw.copy().pick(picks)
raw2.add_channels([raw_ssd])

# -- compute spectrum
psd, freq = mne.time_frequency.psd_welch(raw2, fmin=1, fmax=45, n_fft=3000)

# -- create plot
fig = plt.figure()
outer_grid = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
top_cell = outer_grid[0, :]
gs = gridspec.GridSpecFromSubplotSpec(1, 3, top_cell, width_ratios=[1, 2, 1])

# -- plot time domain
ax1 = plt.subplot(gs[0, 1], zorder=10)
colors = ["k", "k", "#CC2B1C", "#3C2C71"]

raw3 = raw2.copy()
raw3.filter(1, None)

tmin = 89.2
tmax = tmin + 2
raw3.crop(tmin, tmax)

labels = ["electrode 1", "electrode 2", "component 1", "component 2"]

for i in range(len(raw3.ch_names)):
    signal = raw3._data[i]
    signal = signal / np.ptp(signal)
    ax1.plot(raw3.times, signal - 1.2 * i, color=colors[i], lw=1, zorder=20)
    ax1.text(0.0, -1.2 * i - 0.75, labels[i], color=colors[i])
ax1.set(xlim=(0, 2), xlabel="time [s]", yticks=[], ylim=(-4.5, 0.65))

# -- plot spectrum
ax1 = plt.subplot(gs[0, 2])

for i in range(len(raw3.ch_names)):
    ax1.semilogy(freq, psd[i].T, color=colors[i], lw=1)
ax1.set(xlim=(1, 45), xlabel="frequency [Hz]")
ax1.axvspan(
    peak - bin_width, peak + bin_width, color="gray", alpha=0.25, zorder=-3
)
ax1.set(ylabel="log PSD [a.u.]", yticklabels=[])
ax1.yaxis.set_label_position("right")


# -- plot patterns
topo_size = 0.33
coord_y = 0.68
offset_x = 0

for i in range(2):

    ax = plt.Axes(
        fig,
        rect=[offset_x + i * 0.12, coord_y, topo_size, topo_size],
        zorder=1 + 2 * i,
    )
    ax1 = fig.add_axes(ax, zorder=1 + 2 * i)

    im_patterns = helper.make_topoplot(
        patterns[:, i],
        raw.info,
        ax1,
        picks=[picks[i]],
        pick_color=["g"],
        cmap="RdBu_r",
        hemisphere="left",
        plot_head=True,
    )


# -- compile patterns for calculating spatial spread
df = pd.read_csv("../csv/selected_datasets.csv")
df = df[df.is_rest]

participants = df.participant
experiments = df.experiment
df = df.set_index("participant")


nr_patterns = 1
df2 = pd.DataFrame()

for i_s, (participant, exp) in enumerate(zip(participants, experiments)):

    # load electrode positions
    participant_id = "%s_%s" % (participant, exp)
    raw = mne.io.read_raw_fif("../working/%s_raw.fif" % participant_id)
    raw.crop(0, 1)
    raw.load_data()
    raw.pick_types(ecog=True)
    raw = helper.reject_channels(raw, participant, exp)

    # load patterns for all peak frequencies
    peaks = helper.get_participant_peaks(participant, exp)

    for i_p, peak in enumerate(peaks):
        patterns, filters = helper.load_ssd(participant_id, peak)

        for idx in range(nr_patterns):
            pattern = patterns[:, idx]
            xyz = np.array([ich["loc"][:3] for ich in raw.info["chs"]])
            distance = squareform(pdist(xyz, "cityblock"))

            # find maximum value & select distance to maximum
            i_max = np.argmax(np.abs(pattern))
            dist_to_max = distance[i_max]

            # normalize pattern and find maxmium coefficient
            norm_pattern = np.abs(pattern / pattern[i_max])
            idx_dist = (dist_to_max > 0) & (dist_to_max < 25)
            max_coefficent = np.sort(norm_pattern[idx_dist])[-1]

            # save in dataframe
            data = dict(
                spread=max_coefficent,
                participant=participant,
                pattern=pattern,
                peak=peak,
            )
            df2 = df2.append(data, ignore_index=True)


spreads = df2.spread.to_numpy()
freqs = df2.peak.to_numpy()


# -- make plot

top_cell = outer_grid[1, :]
gs = gridspec.GridSpecFromSubplotSpec(
    1, 3, top_cell, width_ratios=[0.4, 1, 0.035]
)

# -- plot spatial spread
ax1 = plt.subplot(gs[0, 0])
ax1.plot(freqs, spreads, "k.", markersize=8, markeredgecolor="#DDDDDD")
ax1.set(
    xlabel="frequency [Hz]",
    ylabel="normalized maximal spatial pattern"
    "\n"
    "coefficient of neighboring electrodes",
    xlim=(4.5, 21),
    xticks=[5, 10, 15, 20],
)
ax1.set_aspect(30)

gs1 = gridspec.GridSpecFromSubplotSpec(2, 3, gs[0, 1])

# -- plot some example topographies with large and small spread
examples_large = (("gc", 12.3), ("jm", 5.4), ("h0", 24.1))
examples_small = (("rr", 12.7), ("mv", 13.5), ("hh", 7.5))
examples = [examples_large, examples_small]

labels = [
    "spatial spread over multiple electrodes",
    "spatial spread with one predominant maximum",
]

for i_cond in range(len(examples)):

    examples1 = examples[i_cond]

    for i in range(len(examples1)):
        ax1 = plt.subplot(gs1[i_cond, i])

        participant, peak = examples1[i]
        df1 = df2[df2.participant == participant]
        df1 = df1[df1.peak == peak]

        pattern = df1.iloc[0].pattern
        participant = df1.iloc[0].participant
        maxC = df1.iloc[0].spread
        peak = df1.iloc[0].peak

        idx_max = np.argmax(np.abs(pattern))
        raw = mne.io.read_raw_fif(
            "../working/%s_fixation_pwrlaw_raw.fif" % participant
        )
        raw.load_data()
        raw.pick_types(ecog=True)
        raw = helper.reject_channels(raw, participant, "fixation_pwrlaw")

        # plot normalized patterns
        pattern = np.abs(pattern)
        pattern = pattern / np.max(pattern)
        sc = helper.make_topoplot(
            pattern,
            raw.info,
            ax1,
            cmap="Reds",
            picks=[idx_max],
            pick_color=["g"],
            vmin=0,
            vmax=1,
            plot_head=None,
            size=40,
        )
        xlim = ax1.get_xlim()
        ax1.set_xlim(xlim[0] - 0.005, xlim[1] + 0.005)
        ax1.set_ylabel("%.2f" % maxC)

        if i == 1:
            ax1.set_title(labels[i_cond])


ax1 = plt.subplot(gs[:, 2])
cb = plt.colorbar(sc, cax=ax1)
cb.set_label("normalized spatial pattern coefficient")
fig.set_size_inches(7.5, 5.5)
fig.savefig("../figures/fig4_spatial_spread.pdf", dpi=300)
fig.show()
