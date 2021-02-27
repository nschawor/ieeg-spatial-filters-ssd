""" Demonstrates noise reduction with spatial filters.
"""
import mne
import numpy as np
import helper
import ssd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_timeseries(raw, ax1, picks, tmin, nr_seconds=2):
    tmax = tmin + nr_seconds
    raw_picks = raw.copy().pick(picks).crop(tmin, tmax)
    for i in range(len(picks)):
        signal = raw_picks._data[i]
        signal = signal / np.ptp(signal)
        ax1.plot(raw_picks.times, signal + i, color="k", lw=0.5)
    ax1.axis("off")


def plot_psd(raw, ax1):
    raw.plot_psd(fmin=2, fmax=220, ax=ax1)
    ax1.set_title("")
    ax1.set_ylabel("")
    ax1.grid(False)
    ax1.set(xticks=[1, 60, 100, 200])
    ax1.set_xlabel("frequency [Hz]")
    ax1.set_yticklabels([])


# -- load continuous data
participant = "jt"
experiment = "faces_basic"
data_file = "../working/%s_%s_raw.fif" % (participant, experiment)
raw = mne.io.read_raw_fif(data_file)
raw.load_data()
raw.pick_types(ecog=True)
raw = helper.reject_channels(raw, participant, experiment)

# -- create common average referenced traces
raw_car = raw.copy()
raw_car.set_eeg_reference("average")

# -- select some channels and and example time point
picks = [86, 17, 19, 21, 1, 50]
tmin = 88

plt.ion()

fig = plt.figure()
gs = gridspec.GridSpec(2, 3)


# -- plot raw traces and spectrum
ax1 = fig.add_subplot(gs[0, 0])
plot_timeseries(raw, ax1, picks, tmin=tmin)
ax1.set_title("raw time series")

ax0 = plt.subplot(gs[1, 0])
plot_psd(raw, ax0)

# -- plot common average referenced traces and spectrum
ax1 = plt.subplot(gs[0, 2])
plot_timeseries(raw_car, ax1, picks, tmin=tmin)
ax1.set_title("common-average\nreferencing")

ax1 = fig.add_subplot(gs[1, 2], sharey=ax0)
plot_psd(raw_car, ax1)


# -- remove noise with SSD, first for 200 Hz
print("estimate noise components...")
peak = 200
bin_width = 1.75
signal_bp = [peak - bin_width, peak + bin_width]
noise_bp = [1, peak + (bin_width + 2)]
noise_bs = [peak - (bin_width + 1), peak + (bin_width + 1)]
filters, patterns = ssd.compute_ssd(raw, signal_bp, noise_bp, noise_bs)

# -- determine number of patterns to remove for 200 Hz
nr_patterns = 3
raw_ssd = ssd.apply_filters(raw, filters[:, :nr_patterns])
noise = patterns[:, :nr_patterns] @ raw_ssd._data[:nr_patterns]
raw._data = raw._data - noise

# -- remove noise with SSD, then for 60 Hz
peak = 60
bin_width = 1.75
signal_bp = [peak - bin_width, peak + bin_width]
noise_bp = [1, peak + (bin_width + 2)]
noise_bs = [peak - (bin_width + 1), peak + (bin_width + 1)]
filters, patterns = ssd.compute_ssd(raw, signal_bp, noise_bp, noise_bs)

# -- determine number of patterns to remove
nr_patterns = 2
raw_ssd = ssd.apply_filters(raw, filters[:, :nr_patterns])
noise = patterns[:, :nr_patterns] @ raw_ssd._data[:nr_patterns]
raw._data = raw._data - noise


# -- plot PSD and time series after SSD removal
ax2 = fig.add_subplot(gs[1, 1], sharey=ax0)
plot_psd(raw, ax2)

ax1 = plt.subplot(gs[0, 1])
plot_timeseries(raw, ax1, picks, tmin=tmin)
ax1.set_title("noise removal with SSD\n for 60 and 200 Hz")

# -- remove tiny topos
topos = [t for t in fig.get_children() if "mpl" in str(type(t))]
[t.remove() for t in topos]

fig.set_size_inches(7.5, 5)
fig.savefig("../figures/fig7_line_noise.pdf", dpi=300)
fig.show()
