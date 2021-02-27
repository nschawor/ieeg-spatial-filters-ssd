""" This script unifies sampling frequency and channel names across datasets
    and saves all provided mat-files into mne fif-file format.
"""
import mne
import numpy as np
import scipy.io
import pandas as pd
import os

df = pd.read_csv("../csv/selected_datasets.csv")
raw_folder = "../working/"
os.makedirs(raw_folder, exist_ok=True)

for i in range(len(df)):

    participant = df.iloc[i].participant
    experiment = df.iloc[i].experiment
    data_file = df.iloc[i].file_name
    print(experiment, participant)

    raw_file = "%s/%s_%s_raw.fif" % (raw_folder, participant, experiment)
    data = scipy.io.loadmat(data_file)

    # extract electrode locations which are either saved together with the data
    # or in a separate file
    if "locs" in data.keys():
        electrodes = data["locs"]
    elif "electrodes" in data.keys():
        electrodes = data["electrodes"]
    elif experiment != "fixation_pwrlaw":
        electrode_file = df.iloc[i].elec_file
        locs = scipy.io.loadmat(electrode_file)
        if "locs" in locs.keys():
            electrodes = locs["locs"]
        elif "electrodes" in locs.keys():
            electrodes = locs["electrodes"]

    # set sampling frequency manually, except where there is a specific entry
    sfreq = 1000
    if participant == "gw":
        sfreq = 10000
    if "srate" in data.keys():
        sfreq = int(data["srate"][0][0])

    ecog = data["data"].T
    nr_ecog_channels = ecog.shape[0]
    ch_names_ecog = ["ecog%i" % i for i in range(nr_ecog_channels)]

    # create stim channel in case there are annotations for events
    if "stim" in data.keys():
        stim = data["stim"].T

        # combine all different types of channels
        data = np.vstack((ecog, stim))
        ch_types = np.hstack((np.tile("ecog", nr_ecog_channels), "stim"))
        channels = ch_names_ecog + ["stim"]
    else:
        data = ecog
        ch_types = np.tile("ecog", nr_ecog_channels)
        channels = ch_names_ecog

    # create montage
    dig_ch_pos = dict(zip(ch_names_ecog, electrodes))
    montage = mne.channels.make_dig_montage(
        ch_pos=dig_ch_pos, coord_frame="head"
    )

    # create raw object
    info = mne.create_info(channels, sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.set_montage(montage)

    # if the data has event-related markers, save them as annotations
    if "stim" in channels:
        events = mne.find_events(raw)

        if experiment == "motor_basic":
            mapping = dict(tongue=11, hand=12)
        if experiment == "faces_basic":
            mapping = dict(isi=101)

        mapping = {v: k for k, v in mapping.items()}

        onsets = events[:, 0] / raw.info["sfreq"]
        durations = np.zeros_like(onsets)  # assumes instantaneous events
        descriptions = [mapping[event_id] for event_id in events[:, 2]]
        annot_from_events = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=raw.info["meas_date"],
        )
        raw.set_annotations(annot_from_events)

    if raw.info["sfreq"] > 1000:
        raw.resample(1000)

    # save raw
    raw.save(raw_file, overwrite=True)
