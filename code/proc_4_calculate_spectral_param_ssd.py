""" This script calculates a spectral parametrization of SSD-filtered spectra.
    The calculation is performed for all participants, the output in terms of
    extracted oscillatory peak parameters (peak frequency and SNR) is saved
    in the form of csv-files.
"""
import mne
import numpy as np
import helper
import ssd
import pandas as pd
import fooof
import os

df = pd.read_csv("../csv/selected_datasets.csv", index_col=False)
df = df[df.is_rest]
participants = df.participant
experiments = df.experiment

nr_seconds = 3
fmax = 55
fmin = 2

results_folder = "../results/spec_param/"
os.makedirs(results_folder, exist_ok=True)

for i_sub, (participant, exp) in enumerate(zip(participants, experiments)):
    print(participant, exp)

    participant_id = "%s_%s" % (participant, exp)
    raw_file = "../working/%s_raw.fif" % participant_id
    raw = mne.io.read_raw_fif(raw_file)
    raw.load_data()
    raw.pick_types(ecog=True)
    raw = helper.reject_channels(raw, participant, exp)

    # get individual peaks
    peaks = helper.get_participant_peaks(participant, exp)
    n_fft = nr_seconds * int(raw.info["sfreq"])
    for peak in peaks:

        print(participant_id, peak)

        df_file_name = "%s/sources_%s_peak_%.2f.csv" % (
            results_folder,
            participant_id,
            peak,
        )

        patterns, filters = helper.load_ssd(participant_id, peak)
        nr_components = np.min([10, patterns.shape[1]])

        raw_ssd = ssd.apply_filters(raw, filters[:, :nr_components])
        psd, freq = mne.time_frequency.psd_welch(
            raw_ssd, fmin=fmin, fmax=fmax, n_fft=n_fft
        )

        fg = fooof.FOOOFGroup(max_n_peaks=5)
        fg.fit(freq, psd)

        peak_params = fg.get_params("peak_params")
        max_peaks = fooof.analysis.periodic.get_band_peak_fg(fg, [fmin, fmax])

        # for each component find electrode with maximum spatial pattern coef
        locations = helper.compute_spatial_max(
            patterns[:, :nr_components], raw, plot=False
        )

        dfs = []
        for i_comp in range(nr_components):
            abs_pattern = np.abs(patterns[:, i_comp])
            idx_chan = np.argmax(abs_pattern)

            max_peak = max_peaks[i_comp][0]
            snr = max_peaks[i_comp, 1]

            data = (
                locations[i_comp, 0],
                locations[i_comp, 1],
                locations[i_comp, 2],
                max_peak,
                snr,
                i_comp,
            )

            df = pd.Series(
                data,
                index=[
                    "x",
                    "y",
                    "z",
                    "freq",
                    "snr",
                    "i_comp",
                ],
            )

            dfs.append(df)
        df = pd.concat(dfs, axis=1).T
        df.to_csv(df_file_name)
