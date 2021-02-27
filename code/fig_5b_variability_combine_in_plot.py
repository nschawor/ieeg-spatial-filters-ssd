""" Combines participant 3d brain plots into figure.
"""
import mne
import numpy as np
import helper
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.image as mpimg
import matplotlib as mpl


df2 = pd.read_csv("../csv/selected_datasets.csv")
df2 = df2[df2.is_rest]

participants = df2.participant
experiments = df2.experiment
results_dir = "../results/spec_param/"
folder_brains = "../figures/3dbrains/"

# -- set parameters
snr_threshold = 0.5
min_freq = 5
max_freq = 20
max_nr_components = 10

# -- plot colorbar; to get this right a dummy figure must be created
peaks_n = np.arange(5, 21, 1)
N = len(peaks_n)
cmap = [plt.cm.viridis(i) for i in np.linspace(0.2, 1, N)]

fig2 = plt.figure()
for i in range(N):
    sc = plt.scatter(i, peaks_n[i], s=10, color=cmap[i])
plt.close(fig2)

# -- create figure
fig = plt.figure()

gs1 = gridspec.GridSpec(
    2,
    2,
    width_ratios=[40, 1],
    height_ratios=[2, 1],
)


ax1 = plt.subplot(gs1[0, 1])
cb = plt.colorbar(sc, cax=ax1)
cb.set_ticks(np.linspace(0, 1, N))
cb.set_ticklabels(peaks_n)
cb.set_label("peak frequency [Hz]")


gs = gridspec.GridSpecFromSubplotSpec(
    5,
    4,
    subplot_spec=gs1[0, 0],
)


# -- determine order of participants according to electrode mean y-position
mean_loc = np.zeros((len(participants), 3))
for i_sub, (participant, exp) in enumerate(zip(participants, experiments)):
    participant_id = "%s_%s" % (participant, exp)
    file_name = "../working/%s_raw.fif" % participant_id
    raw = mne.io.read_raw_fif(file_name)

    electrodes = np.array([e["loc"][:3] for e in raw.info["chs"]])
    mean_loc[i_sub] = electrodes.mean(axis=0)


idx = np.argsort(mean_loc[:, 2])[::-1]
participants = participants[idx]
experiments = experiments[idx]

# -- plot small 3d brains
for i_s, (participant, exp) in enumerate(zip(participants, experiments)):
    file_name = "%s/localization_%s_%s_spatial_max.png" % (
        folder_brains,
        participant,
        exp,
    )

    ax1 = plt.subplot(gs[i_s])
    img = mpimg.imread(file_name)
    ax1.imshow(img)
    ax1.set(yticks=[], xticks=[])
    ax1.axis("off")


# -- plot peak frequencies across space for each participant

norm = mpl.colors.Normalize(vmin=-65.0, vmax=65.0)
N = len(df2)
colors = [plt.cm.cividis(i) for i in np.linspace(0, 1, N)]

gs = gridspec.GridSpecFromSubplotSpec(
    1,
    3,
    subplot_spec=gs1[1, :],
    width_ratios=[3, 1, 0.1],
    wspace=0.2,
    hspace=0.0,
)


# -- collect peak frequency information
data = []
for i_sub, (participant, exp) in enumerate(zip(participants, experiments)):
    participant_id = "%s_%s" % (participant, exp)

    df1 = df2[df2.participant == participant]
    peaks = helper.get_participant_peaks(participant, exp)

    for peak in peaks:

        df = pd.read_csv(
            "%s/sources_%s_peak_%.2f.csv" % (results_dir, participant_id, peak)
        )
        df = df[df.snr > snr_threshold]
        df = df[df.freq < max_freq]
        nr_components = np.min([max_nr_components, len(df)])

        for i_comp in range(nr_components):
            location = df.iloc[i_comp][["x", "y", "z"]].to_numpy()
            max_peak = df.iloc[i_comp]["freq"]
            snr = df.iloc[i_comp]["snr"]

            data.append((i_sub, max_peak, snr, location[1]))


# -- plot pooled peak frequencies
ax = plt.subplot(gs[0, 0])
yticks = [5, 10, 15, 20]

# -- plot markers for all components
data = np.array(data)
all_peaks = data[:, 1]
sc = ax.scatter(
    data[:, 0],
    data[:, 1],
    s=30 * data[:, 2],
    c=mpl.cm.plasma(norm(data[:, 3])),
    alpha=0.75,
)

ax.set(
    xlabel="participant",
    xticklabels=[],
    xticks=range(0, N + 1, 1),
    yticks=yticks,
    ylim=(4.5, 20.5),
    ylabel="peak frequency [Hz]",
)


ax2 = plt.subplot(gs[0, 2])
cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.plasma), cax=ax2)
cb.set_label("component location along\nposterior-anterior axis")

# -- histogram of pooled peak frequencies
ax1 = plt.subplot(gs[0, 1], sharey=ax)
for i in yticks:
    ax1.axhline(i, linestyle="dotted", color="red", alpha=0.2)
bins = np.linspace(0, 25, 1)
ax1.hist(
    np.array(all_peaks), 30, facecolor="k", alpha=0.5, orientation="horizontal"
)
ax1.set_ylabel("peak frequency [Hz]")
ax1.yaxis.set_ticks_position("left")
ax1.yaxis.set_label_position("right")

ax1.yaxis.tick_left()
ax1.set_xticklabels([])
ax1.set(yticks=yticks, ylim=(4.5, 20.5))

fig.set_size_inches(7.5, 8.5)
fig.savefig("../figures/fig5_variability.pdf", dpi=300)
fig.show()
