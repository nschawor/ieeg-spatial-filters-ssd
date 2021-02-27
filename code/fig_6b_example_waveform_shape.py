""" Shows example of close-by rhythms with different waveform.
"""
import mne
import numpy as np
import matplotlib.pyplot as plt
import helper
import ssd
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.image as mpimg

plt.ion()

# -- load continuous data
participant_id = "jc_fixation_pwrlaw"
raw = mne.io.read_raw_fif("../working/%s_raw.fif" % participant_id)
raw.load_data()
raw.pick_types(ecog=True)

peak = 9.45
bin_width = 1.75
filters, patterns = ssd.run_ssd(raw, peak, bin_width)
nr_components = 2
raw_ssd = ssd.apply_filters(raw, filters[:, :nr_components])


raw_ssd.filter(2, None)
raw.filter(2, None)

# -- pick the channels contributing most to the spatial patterns for comparison
picks = [np.argmax(np.abs(patterns[:, 0])), np.argmax(np.abs(patterns[:, 1]))]
raw2 = raw.copy().pick(picks)

raw_ssd2 = raw_ssd.copy().pick_channels(raw_ssd.ch_names[:2])
raw_ssd2.add_channels([raw2])
raw2 = raw_ssd2.copy()

# -- compute spectrum
psd, freq = mne.time_frequency.psd_welch(raw2, fmin=1, fmax=45, n_fft=3000)

fig = plt.figure()
gs = gridspec.GridSpec(
    2,
    4,
    width_ratios=[3, 1, 0.9, 0.75],
)

# -- plot time domain
ax1 = plt.subplot(gs[0, 0])

colors = ["#CC2B1C", "#3C2C71", "k", "k"]
raw3 = raw2.copy()
raw3.filter(1, None)

tmin = 37.25
tmax = tmin + 2.5
raw3.crop(tmin, tmax)

labels = [
    "comp.1",
    "comp.2",
    "e1",
    "e2",
]

signs = [1, 1, 1, 1, 1]
for i in range(len(raw3.ch_names)):
    signal = signs[i] * raw3._data[i]
    signal = signal / np.ptp(signal)
    ax1.plot(raw3.times, signal - 1.05 * i, color=colors[i], lw=1)
    ax1.text(-0.05, -1 * i, labels[i], color=colors[i], ha="right", va="center")

ax1.set(xlim=(0, 1.75), xlabel="time [s]", yticks=[], xticks=[0, 1, 2])

# -- plot spectrum
ax1 = plt.subplot(gs[0, 1:3])

for i in range(len(raw3.ch_names)):
    ax1.semilogy(freq, psd[i].T, color=colors[i], lw=1)
ax1.set(
    xlim=(1, 45),
    xlabel="frequency [Hz]",
)
ax1.axvspan(
    peak - bin_width, peak + bin_width, color="gray", alpha=0.25, zorder=-3
)

# -- plot peak frequency markers
ax1.axvline(peak, color="tab:green", lw=1, zorder=-2)
ax1.axvline(2 * peak, color="tab:green", lw=1, zorder=-2)
ax1.axvline(3 * peak, color="tab:green", lw=1, zorder=-2)


# -- plot participant summary
ax1 = plt.subplot(gs[1, 0])

df = pd.read_csv("../csv/selected_datasets.csv")
df = df[df.is_rest]
nr_participants = len(df)
results_folder = "../results/bursts/"

counter = 0
nr_bursts = []
asymmetry = []
for i_sub, (participant, exp) in enumerate(zip(df.participant, df.experiment)):

    participant_id = "%s_%s" % (participant, exp)
    peaks = helper.get_participant_peaks(participant, exp)

    for peak1 in peaks:
        df_file_name = "%s/bursts_%s_peak_%s.csv" % (
            results_folder,
            participant_id,
            peak1,
        )

        df = pd.read_csv(df_file_name)
        df = df.set_index("feature")

        if df.value["nr_bursts"] == 0:
            continue

        df = df.T
        frequency = 1000 / df.period
        if np.any(frequency < 5):
            continue
        ax1.plot(
            frequency,
            2 * np.abs(df.time_ptsym - 0.5),
            ".",
            markeredgecolor="w",
            markerfacecolor="k",
            markersize=12,
        )

        nr_bursts.append(df["nr_bursts"].to_list())
        asymmetry.append(2 * np.abs(df.time_ptsym.to_list()[0] - 0.5))

        counter += 1

ax1.set(
    xlabel="peak frequency [Hz]",
    ylabel="peak-trough asymmetry",
    xlim=(4.5, 20.5),
    xticks=(range(5, 21, 5)),
)

# -- asymmetries across 3d brain
ax1 = plt.subplot(gs[1, 1:])
plot_filename = "../figures/3dbrain_asymmetry.png"
img = mpimg.imread(plot_filename)

ax1.imshow(img)
ax1.set_xlim(25, 1000)
ax1.set_ylim(725, 0)
ax1.set(yticks=[], xticks=[])
ax1.axis("off")

peaks_n = np.arange(0, 0.21, 0.01)
N = len(peaks_n)
cmap = [plt.cm.plasma(i) for i in np.linspace(0.2, 1, N)]


fig2 = plt.figure()
sc = plt.scatter(
    range(N), range(N), c=peaks_n, s=5, vmin=0, vmax=0.2, cmap=plt.cm.plasma
)
plt.show()
fig2.show()

cb = plt.colorbar(
    sc, ax=ax1, orientation="horizontal", fraction=0.05, shrink=0.8, pad=0.0
)
cb.set_label("peak-trough asymmetry")
plt.close(fig2)


# -- plot patterns
topo_size = 0.18

for i in range(2):

    ax = plt.Axes(fig, rect=[0.8, 0.77 - i * 0.2, topo_size, topo_size])
    ax1 = fig.add_axes(ax)

    helper.make_topoplot(
        signs[i] * patterns[:, i],
        raw.info,
        ax1,
        picks=[picks[i]],
        cmap1=["g"],
        cmap="RdBu_r",
        side="left",
        plot_head=None,
        edgecolor="tab:green",
    )


fig.set_size_inches(7.5, 6)
fig.savefig("../figures/fig6_example_waveform.pdf", dpi=300)
fig.show()
