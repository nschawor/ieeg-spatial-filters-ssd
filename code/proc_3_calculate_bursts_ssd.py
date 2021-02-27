""" This script performs burst detection across all resting datasets.
    The same burst detection parameters are used across datasets. The output
    is saved in the form of csv-files.
"""

import mne
import helper
import ssd
import pandas as pd
import os
from bycycle.features import compute_features

df = pd.read_csv("../csv/selected_datasets.csv", index_col=False)
df = df[df.is_rest]

participants = df.participant
experiments = df.experiment

results_folder = "../results/bursts/"
os.makedirs(results_folder, exist_ok=True)

features = [
    "period",
    "time_trough",
    "time_peak",
    "volt_trough",
    "volt_peak",
    "time_rise",
    "time_decay",
    "volt_rise",
    "volt_decay",
    "volt_amp",
    "time_rdsym",
    "time_ptsym",
]

# -- setting the same burst detection parameters for all datasets
osc_param = {
    "amplitude_fraction_threshold": 0.75,
    "amplitude_consistency_threshold": 0.5,
    "period_consistency_threshold": 0.5,
    "monotonicity_threshold": 0.5,
    "N_cycles_min": 3,
}

freq_width = 3

for i_sub, (participant, exp) in enumerate(zip(participants, experiments)):
    print(participant, exp)
    dfs = []

    participant_id = "%s_%s" % (participant, exp)
    raw_file = "../working/%s_raw.fif" % participant_id
    raw = mne.io.read_raw_fif(raw_file)
    raw.load_data()
    raw.pick_types(ecog=True)
    raw = helper.reject_channels(raw, participant, exp)

    # get individual peaks
    peaks = helper.get_participant_peaks(participant, exp)

    for peak1 in peaks:

        patterns, filters = helper.load_ssd(participant_id, peak1)
        raw_ssd = ssd.apply_filters(raw, filters[:, :2])
        bandwidth = (float(peak1) - freq_width, float(peak1) + freq_width)
        i_comp = 0

        # compute features for SSD component
        df = compute_features(
            raw_ssd._data[i_comp],
            raw.info["sfreq"],
            f_range=bandwidth,
            burst_detection_kwargs=osc_param,
            center_extrema="T",
        )

        # save mean burst features
        df = df[df.is_burst]
        nr_bursts = len(df)
        df1 = df.mean()
        df1 = df1[features]
        df1["comp"] = i_comp
        df1["nr_bursts"] = nr_bursts

        df_file_name = "%s/bursts_%s_peak_%s.csv" % (
            results_folder,
            participant_id,
            peak1,
        )

        df1 = df1.reset_index()
        df1.to_csv(df_file_name, header=["feature", "value"], index=False)
