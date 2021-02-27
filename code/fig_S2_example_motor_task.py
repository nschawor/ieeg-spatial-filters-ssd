""" This figure shows an example of SSD referenced activity for a task.
"""
import mne
import numpy as np
import ssd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from bycycle.features import compute_features

plt.ion()

# -- load continuous data
exp_id = "motor_basic"
participant = "ug"
file_name = "../working/%s_%s_raw.fif" % (participant, exp_id)
raw = mne.io.read_raw_fif(file_name, preload=True)
raw.pick_types(ecog=True)

# -- get common average referenced activity
nr_channels = len(raw.ch_names)
filters_car = np.zeros((nr_channels, nr_channels)) - 1 / nr_channels
for i in range(nr_channels):
    filters_car[i, i] = 1
raw_car = ssd.apply_filters(raw, filters_car, prefix="car")

# -- compute SSD
bin_width = 1.2
peak = 9.46
filters, patterns = ssd.run_ssd(raw, peak, bin_width)
patterns_car = filters_car.T @ patterns

# -- get SSS referenced signal
nr_components = 3
raw_ssd = ssd.apply_filters(raw, filters[:, :nr_components])

# -- detect bursts with bycycle
picks = [np.argsort(np.abs(patterns_car[:, 0]))[-1]]
nr_trials = 15
raw_car.pick(picks)

osc_param = {
    "amplitude_fraction_threshold": 0.5,
    "amplitude_consistency_threshold": 0.5,
    "period_consistency_threshold": 0.5,
    "monotonicity_threshold": 0.5,
    "N_cycles_min": 3,
}

bins = np.linspace(0, 1, 21)
bandwidth = (peak - 2, peak + 2)
Fs = int(raw.info["sfreq"])


raw_car.filter(3, None)
raw_ssd.filter(3, None)

i_comp = 0
df_ssd = compute_features(
    raw_ssd._data[i_comp],
    Fs,
    f_range=bandwidth,
    burst_detection_kwargs=osc_param,
    center_extrema="P",
)
df_ssd = df_ssd[df_ssd.is_burst]

i = 0
df = compute_features(
    raw_car._data[i],
    Fs,
    f_range=bandwidth,
    burst_detection_kwargs=osc_param,
    center_extrema="P",
)
df = df[df.is_burst]


# -- cut continuous signal into epochs
events, event_id = mne.events_from_annotations(raw)
tmin = -1.5
tmax = 2.5

colors = plt.cm.tab20b(np.linspace(0, 1, 20))

nr_trials = 15
epochs = mne.Epochs(raw_ssd, events, tmin=tmin, tmax=tmax)
epochs.load_data()

# -- create figure
fig = plt.figure()
gs = gridspec.GridSpec(2, 2)

labels = ["hand movement\nSSD", "tongue movement\nSSD"]
for i_type, type in enumerate(["1", "2"]):
    counter = 0
    ax = plt.subplot(gs[0, i_type])
    epochs1 = epochs[type]

    idx = events[:, 2] == int(type)
    events1 = events[idx]

    for i in range(nr_trials):

        sample = events1[i, 0]

        for ii in range(1):

            data = epochs1._data[i, ii]
            data = data / np.ptp(data)
            ax.plot(
                epochs.times, data - counter, color=colors[2 * i_type], lw=0.6
            )

            idx_trial = (df_ssd["sample_peak"] > sample + tmin * Fs) & (
                df_ssd["sample_peak"] < sample + tmax * Fs
            )

            df1 = df_ssd[idx_trial]

            starts = raw.times[df1.sample_last_trough] - raw.times[sample]
            ends = raw.times[df1.sample_next_trough] - raw.times[sample]

            for iii in range(len(df1)):
                ax.fill(
                    [starts[iii], starts[iii], ends[iii], ends[iii]],
                    [
                        -counter - 0.5,
                        -counter + 0.5,
                        -counter + 0.5,
                        -counter - 0.5,
                    ],
                    color=colors[2 * i_type],
                    alpha=0.2,
                )

        counter -= 1
    ax.axvline(0, color="k")
    ax.set(
        xlim=(tmin, tmax),
        title=labels[i_type],
        xlabel="time relative to movement cue [s]",
        ylabel="trial number",
        yticks=np.arange(nr_trials),
        yticklabels=np.arange(nr_trials)[::-1] + 1,
    )

    ax.set_ylim(-0.5, nr_trials)


# -- plot common average referenced signal
epochs = mne.Epochs(raw_car, events, tmin=tmin, tmax=tmax)
epochs.load_data()
labels = [
    "hand movement\ncommon average reference",
    "tongue movement\ncommon average reference",
]

for i_type, type in enumerate(["1", "2"]):
    counter = 0
    ax = plt.subplot(gs[1, i_type])
    epochs1 = epochs[type]

    idx = events[:, 2] == int(type)
    events1 = events[idx]

    for i in range(nr_trials):

        sample = events1[i, 0]

        for ii in range(1):
            data = epochs1._data[i, ii]
            data = data / np.ptp(data)
            ax.plot(
                epochs.times,
                data - counter,
                color=colors[2 * i_type + 12],
                lw=0.6,
            )

            idx_trial = (df["sample_peak"] > sample + tmin * Fs) & (
                df["sample_peak"] < sample + tmax * Fs
            )

            df1 = df[idx_trial]

            starts = raw.times[df1.sample_last_trough] - raw.times[sample]
            ends = raw.times[df1.sample_next_trough] - raw.times[sample]
            for iii in range(len(df1)):
                ax.fill(
                    [starts[iii], starts[iii], ends[iii], ends[iii]],
                    [
                        -counter - 0.5,
                        -counter + 0.5,
                        -counter + 0.5,
                        -counter - 0.5,
                    ],
                    color=colors[2 * i_type + 12],
                    alpha=0.2,
                )

        counter -= 1
    ax.axvline(0, color="k")
    ax.set(
        xlim=(tmin, tmax),
        yticks=np.arange(nr_trials),
        yticklabels=np.arange(nr_trials)[::-1] + 1,
        title=labels[i_type],
        xlabel="time relative to movement cue [s]",
        ylabel="trial number",
    )
    ax.set_ylim(-0.5, nr_trials)


fig.set_size_inches(7.5, 8)
fig.savefig("../figures/fig_S2_bursts_example.pdf", dpi=300)
fig.show()
