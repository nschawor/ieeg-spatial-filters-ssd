""" This script calculates a spectral parametrization of power spectra.
    The calculation is performed for all electrodes, to inform selection of
    peak frequencies for calculation of data-driven spatial filters.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import helper
import pandas as pd
import fooof
import os

df_sub = pd.read_csv("../csv/selected_datasets.csv", index_col=False)
df_sub = df_sub[df_sub.is_rest]

participants = df_sub.participant
experiments = df_sub.experiment
df_sub = df_sub.set_index("participant")
nr_seconds = 3
fmax = 55
fmin = 2


for i_sub, (participant, exp) in enumerate(zip(participants, experiments)):
    print(participant, exp)
    dfs = []

    participant_id = "%s_%s" % (participant, exp)
    raw_file = "../working/%s_raw.fif" % participant_id
    raw = mne.io.read_raw_fif(raw_file)
    raw.load_data()
    raw.pick_types(ecog=True)
    raw = helper.reject_channels(raw, participant, exp)

    psd, freq = mne.time_frequency.psd_welch(
        raw,
        fmin=fmin,
        fmax=fmax,
        n_fft=nr_seconds * int(raw.info["sfreq"]),
    )

    # calculate spectral parametrization for each channel
    plot_folder = "../results/psd_param/%s/" % participant
    os.makedirs(plot_folder, exist_ok=True)

    # always plot 9 electrodes together in a figure for saving some space
    fig, ax = plt.subplots(3, 3)
    counter = 0
    i_fig = 0
    for i in range(len(raw.ch_names)):
        ax1 = ax.flatten()[counter]
        fg = fooof.FOOOF(max_n_peaks=5, min_peak_height=0.5)
        fg.fit(freq, psd[i])
        fg.plot(
            ax=ax1,
            add_legend=False,
            plot_style=None,
            plot_peaks="line",
        )

        # if peaks were identified print them into axes
        peak_params = fg.get_params("peak_params")
        if not (np.isnan(peak_params.flatten()[0])):
            ax1.text(
                15,
                ax1.get_ylim()[1] - 0.05 * ax1.get_ylim()[1],
                peak_params,
                va="top",
                color="tab:green",
                fontsize=10,
            )
        ax1.set_title(raw.ch_names[i])
        ax1.set(xlim=(fmin, fmax), xlabel="frequency [Hz]", yticklabels=[])
        counter += 1
        if counter == 9:
            fig.set_size_inches(8, 8)
            fig.tight_layout()
            fig.savefig("%s/spec_param_%i.png" % (plot_folder, i_fig))
            fig.show()
            plt.close("all")
            fig, ax = plt.subplots(3, 3)
            counter = 0
            i_fig += 1
