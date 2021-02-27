""" This script computes spatial filters and patterns.
    This procedure is done for all participants across selected peak
    frequencies. The filters and patterns are saved as numpy arrays.
"""
import mne
import os
import numpy as np
import helper
import ssd
import pandas as pd

df = pd.read_csv("../csv/selected_datasets.csv", index_col=False)
df = df[df.is_rest]

participants = df.participant
experiments = df.experiment
df = df.set_index("participant")

results_dir = "../results/ssd/"
os.makedirs(results_dir, exist_ok=True)


for participant, experiment in list(zip(participants, experiments)):

    participant_id = "%s_%s" % (participant, experiment)
    print(participant_id)

    # get individual peaks
    peaks, bin_width1 = helper.get_participant_peaks(
        participant, experiment, return_width=True
    )

    # load raw-file
    file_name = "../working/%s_raw.fif" % participant_id
    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.pick_types(ecog=True)
    raw = helper.reject_channels(raw, participant, experiment)

    for i_peak, peak in enumerate(peaks):

        file_name = "%s/ssd_%s_peak_%.2f.npy" % (
            results_dir,
            participant_id,
            peak,
        )
        bin_width = bin_width1[i_peak]

        filters, patterns = ssd.run_ssd(raw, peak, bin_width)

        data_dict = dict(peak=peak, filters=filters, patterns=patterns)
        np.save(file_name, data_dict)
