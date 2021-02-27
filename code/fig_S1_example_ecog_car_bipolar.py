""" This figure shows examples of different spatial filters for 1 participant.

"""
import mne
import numpy as np
import helper
import ssd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_timeseries(
    ax1, raw, cmap=["#2d004f", "#254f00", "#000000"], label=None
):

    nr_channels = len(raw.ch_names)
    for i_pick in range(nr_channels):
        signal = raw._data[i_pick]
        signal = signal / np.ptp(signal)
        ax1.plot(raw.times, signal - i_pick, color=cmap[i_pick], lw=1)

    ymin = -nr_channels + 0.25
    ymax = 0.45
    ax1.set(
        xlim=(raw.times[0], raw.times[-1]),
        xticks=np.arange(0, 2.01, 0.5),
        yticks=[],
        ylim=(ymin, ymax),
        title=label,
    )


def plot_contribution(ax, contributions):

    vmax = np.max(np.abs(contributions))

    ax.imshow(contributions, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
    ax.plot(np.arange(-0.5, 3, 1), 0.49 * np.ones((4,)), color="k", lw=1)
    ax.plot(np.arange(-0.5, 3, 1), 1.49 * np.ones((4,)), color="k", lw=1)
    ax.set(yticks=[])
    ax.set_xticks([0, 1, 2])
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticklabels(["comp.%i" % (i + 1) for i in range(3)], rotation=45)

    for i in range(3):
        ax.get_xticklabels()[i].set_color(cmap[i])

    for (j, i), label in np.ndenumerate(contributions):
        ax.text(i, j, "%.2f" % label, ha="center", va="center")


# -- load continuous data
exp_id = "motor_basic"
participant = "ug"
file_name = "../working/%s_%s_raw.fif" % (participant, exp_id)
raw = mne.io.read_raw_fif(file_name, preload=True)
raw.pick_types(ecog=True)

# -- set parameters
bin_width = 1.2
peak = 9.46
nr_channels = len(raw.ch_names)
nr_seconds = 3
chans_to_plot = 3
n_fft = int(nr_seconds * raw.info["sfreq"])
tmin = 4
tmax = tmin + 2


# -- create plot
fig = plt.figure()
gs = gridspec.GridSpec(
    4,
    3,
    width_ratios=[1.25, 2.5, 1],
)

# -- apply SSD
filters, patterns_ssd = ssd.run_ssd(raw, peak, bin_width)
nr_components = 3
raw_ssd = ssd.apply_filters(raw, filters[:, :nr_components])
SNR_ssd = helper.get_SNR(raw_ssd)

signs = [1, -1, -1]
for i_s, sign in enumerate(signs):
    raw_ssd._data[i_s] *= sign
    patterns_ssd[:, i_s] *= sign


cmap = [plt.cm.viridis(i) for i in np.linspace(0.2, 1, 4)]
ax1 = plt.subplot(gs[3, 0])
helper.plot_psd(
    ax1, raw_ssd, cmap=cmap, SNR=SNR_ssd, peak=peak, bin_width=bin_width
)

ax1 = plt.subplot(gs[3, 1])
raw_ssd.crop(tmin, tmax)
plot_timeseries(ax1, raw_ssd, cmap=cmap, label="SSD")
ax1.set_xlabel("time [ms]")

# -- make bipolar anterior to posterior
rows = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24],
]


raw_AP, filters_AP, elecs = helper.create_bipolar_derivation(
    raw, rows, prefix="AP"
)
SNR_AP = helper.get_SNR(raw_AP)

picks = np.argsort(SNR_AP)[-chans_to_plot:][::-1]
SNR_AP = SNR_AP[picks]
elecs = elecs[picks]
raw_AP.pick(list(picks))


# -- plot power spectrum
cmap = ["#470509", "#76181E", "#A6373F"]
ax1 = plt.subplot(gs[0, 0])
helper.helper.plot_psd(
    ax1, raw_AP, SNR=SNR_AP, peak=peak, cmap=cmap, bin_width=bin_width
)

ax1 = plt.subplot(gs[0, 1])
raw_AP.crop(tmin, tmax)
plot_timeseries(ax1, raw_AP, label="bipolar posterior-anterior", cmap=cmap)

ax1 = plt.subplot(gs[0, 2])
helper.make_topoplot(
    np.zeros(
        (nr_channels,),
    ),
    raw.info,
    ax=ax1,
    picks=elecs,
    cmap="Greys",
    plot_head=None,
    cmap1=cmap,
)


# -- make bipolar medial - lateral
rows = [
    [0, 5, 10, 15, 20],
    [1, 6, 11, 16, 21],
    [2, 7, 12, 17, 22],
    [3, 8, 13, 18, 23],
    [4, 9, 14, 19, 24],
]


raw_ML, filters_ML, elecs = helper.create_bipolar_derivation(
    raw, rows, prefix="ML"
)
SNR_ML = helper.get_SNR(raw_ML)

picks = np.argsort(SNR_ML)[-chans_to_plot:][::-1]
SNR_ML = SNR_ML[picks]
elecs = elecs[picks]


raw_ML.pick(list(picks))

cmap = ["#0E1746", "#313B74", "#49548B"]

ax1 = plt.subplot(gs[1, 0])
helper.plot_psd(
    ax1, raw_ML, SNR=SNR_ML, peak=peak, cmap=cmap, bin_width=bin_width
)

ax1 = plt.subplot(gs[1, 1])
raw_ML.crop(tmin, tmax)
plot_timeseries(ax1, raw_ML, label="bipolar medial-lateral", cmap=cmap)

# -- highlight electrodes
ax1 = plt.subplot(gs[1, 2])
helper.make_topoplot(
    np.zeros(
        (nr_channels,),
    ),
    raw.info,
    ax=ax1,
    picks=elecs,
    cmap="Greys",
    plot_head=None,
    cmap1=cmap,
)

# -- common average reference

filters_car = np.zeros((nr_channels, nr_channels)) - 1 / nr_channels
for i in range(nr_channels):
    filters_car[i, i] = 1

raw_car = ssd.apply_filters(raw, filters_car, prefix="car")
SNR_car = helper.get_SNR(raw_car)

picks = np.argsort(SNR_car)[-chans_to_plot:][::-1]
SNR_car = SNR_car[picks]
raw_car.pick(picks)

ax1 = plt.subplot(gs[2, 0])
cmap = ["#034500", "#156711", "#5AAC56"]
helper.plot_psd(
    ax1, raw_car, SNR=SNR_car, peak=peak, cmap=cmap, bin_width=bin_width
)

ax1 = plt.subplot(gs[2, 1])
raw_car.crop(tmin, tmax)
plot_timeseries(ax1, raw_car, label="common average", cmap=cmap)
ax1.axvspan(0.75, 1.5, ymin=0.66, ymax=0.99, edgecolor="r", facecolor=None)
ax1 = plt.subplot(gs[2, 2])
helper.make_topoplot(
    np.zeros(
        (nr_channels,),
    ),
    raw.info,
    ax=ax1,
    picks=picks,
    cmap="Greys",
    plot_head=None,
    cmap1=cmap,
)


fig.set_size_inches(7.5, 7.5)
fig.savefig("../figures/fig_S1_example_car.pdf", dpi=300)
fig.show()
