""" This figure shows an example of SSD application for one participant.
"""
import mne
import numpy as np
import helper
import ssd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


plt.ion()
exp_id = "motor_basic"
participant = "ug"

# -- load continuous data
file_name = "../working/%s_%s_raw.fif" % (participant, exp_id)
raw = mne.io.read_raw_fif(file_name, preload=True)
raw.pick_types(ecog=True)
raw_org = raw.copy()

# -- apply SSD and compute SNR for all components
bin_width = 1.2
peak = 9.46
nr_components = 3

filters, patterns = ssd.run_ssd(raw_org, peak, bin_width)
raw_ssd = ssd.apply_filters(raw_org, filters)

# -- SSD is polarity invariant, align to electrode signals by visual inspection
signs = [1, -1, -1]
for i_s, sign in enumerate(signs):
    patterns[:, i_s] *= sign
    raw_ssd._data[i_s] *= sign

SNR_ssd = helper.get_SNR(raw_ssd, freq=[peak - 2, peak + 2])
raw_ssd.pick(range(nr_components))

# -- select electrodes and compute SNR for the selected peak frequency
ch_names = ["ecog16", "ecog10", "ecog20"]
picks = mne.pick_channels(raw.ch_names, ch_names, ordered=True)
raw.pick(picks)
SNR_electrodes = helper.get_SNR(raw, freq=[peak - 2, peak + 2])

# -- create figure
fig = plt.figure()
outer_grid = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1])
top_cell = outer_grid[0, :]
bottom_cell = outer_grid[1, :]
gs = gridspec.GridSpecFromSubplotSpec(
    2, 3, top_cell, width_ratios=[1.25, 2.5, 1]
)

# -- plot electrode PSD + SNR
ax1 = plt.subplot(gs[0, 0])
cmap1 = ["#2d004f", "#254f00", "#000000"]
helper.plot_psd(
    ax1,
    raw,
    cmap=cmap1,
    SNR=SNR_electrodes,
    peak=peak,
    bin_width=bin_width,
)

# -- plot PSD for SSD component
ax1 = plt.subplot(gs[1, 0])
cmap = [plt.cm.viridis(i) for i in np.linspace(0.2, 1, 4)]

helper.plot_psd(
    ax1, raw_ssd, cmap=cmap, SNR=SNR_ssd, peak=peak, bin_width=bin_width
)

# -- plot time domain signals
tmin = 4
tmax = tmin + 2

raw_ssd.filter(2, None)
raw.filter(2, None)

raw.crop(tmin, tmax)
raw_ssd.crop(tmin, tmax)

# -- plot electrode signals
ax1 = plt.subplot(gs[0, 1])
helper.plot_timeseries(ax1, raw, cmap=cmap1, label="")
ax1.set_xlabel("time [ms]")


# -- pattern coefficients
ax_cont = plt.subplot(gs[0, 2])

contributions = patterns[picks, :nr_components]
vmax = np.max(np.abs(contributions))
ax_cont.imshow(contributions, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
ax_cont.set(yticks=[])
ax_cont.set_xticks(range(nr_components))
ax_cont.set_xticklabels(["comp.%i" % (i + 1) for i in range(nr_components)])


for (j, i), label in np.ndenumerate(contributions):
    ax_cont.text(i, j, "%.2f" % label, ha="center", va="center")


# -- plot SSD time series
ax1 = plt.subplot(gs[1, 1])
helper.plot_timeseries(ax1, raw_ssd, cmap=cmap, label="")
ax1.set_xlabel("time [ms]")
for i_chan in range(nr_components):
    ax1.text(0, -i_chan + 0.35, "component %i" % (i_chan + 1))


# -- plot SNR for SSD components
ax1 = plt.subplot(gs[1, 2])
ax1.plot(SNR_ssd, ".-", color="k", markeredgecolor="w", markersize=8)
for i in range(nr_components):
    ax1.plot(
        i, SNR_ssd[i], ".", color=cmap[i], markeredgecolor="w", markersize=12
    )
ax1.set(xlabel="component number", ylabel="SNR [dB]")


# -- plot filters
gs2 = gridspec.GridSpecFromSubplotSpec(1, 7, bottom_cell)
topo_size = 0.26

for i in range(nr_components):

    ax = plt.Axes(fig, rect=[i * 0.18 + 0.35, 0.12, topo_size, topo_size])
    ax1 = fig.add_axes(ax)

    im_filters = helper.make_topoplot(
        filters[:, i],
        raw_org.info,
        ax,
        plot_head=False,
        picks=picks,
        cmap="PiYG",
        pick_color=["dimgrey"],
        vmin=-0.75,
        vmax=0.75,
    )

    ax1.set_ylim(-0.04, 0.04)
    ax1.set_xlim(-0.08, 0.08)

# -- plot patterns
for i in range(nr_components):

    ax = plt.Axes(fig, rect=[i * 0.18 + 0.35, -0.05, topo_size, topo_size])
    ax1 = fig.add_axes(ax)

    im_patterns = helper.make_topoplot(
        patterns[:, i],
        raw_org.info,
        ax1,
        plot_head=False,
        picks=picks,
        cmap="RdBu_r",
        vmin=-1.5,
        vmax=1.5,
        pick_color=["dimgrey"],
    )

    ax1.set_title("component %i" % (i + 1))
    ax1.set_ylim(-0.04, 0.04)
    ax1.set_xlim(-0.08, 0.08)


# -- filters & patterns colorbars
ax = plt.Axes(fig, rect=[0.3, 0.225, 0.15, 0.025])
ax1 = fig.add_axes(ax)
cb = plt.colorbar(im_filters, cax=ax1, orientation="horizontal")
ax1.set_title("spatial filters")

ax = plt.Axes(fig, rect=[0.3, 0.05, 0.15, 0.025])
ax1 = fig.add_axes(ax)
cb = plt.colorbar(im_patterns, cax=ax1, orientation="horizontal")
ax1.set_title("spatial patterns")

# -- electrodes on topo head
ax = plt.Axes(fig, rect=[0.0, 0.02, topo_size, topo_size])
ax1 = fig.add_axes(ax)
mask = np.zeros((len(raw_org.ch_names)), dtype="bool")
mask[picks] = True
mne.viz.plot_topomap(
    np.zeros((len(raw_org.ch_names),)) + np.nan,
    raw_org.info,
    axes=ax1,
    mask=mask,
)

fig.set_size_inches(7.5, 6)
fig.savefig("../figures/fig2_example_ecog.pdf", dpi=300)
fig.show()
