""" Creates 3d brain + all components with SNR > threshold plots for all
    participants.
"""
import mne
import numpy as np
import matplotlib.pyplot as plt
import helper
import pandas as pd
import pyvista as pv
import os

df = pd.read_csv("../csv/selected_datasets.csv", index_col=False)
df = df[df.is_rest]

participants = df.participant
experiments = df.experiment

nr_components = 10
nr_seconds = 5
results_folder = "../results/spec_param/"
plot_folder = "../figures/3dbrains/"

os.makedirs(results_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

# -- for creating peak frequency colorbar
peaks_n = np.arange(5, 21, 1)
N = len(peaks_n)
cmap = [plt.cm.viridis(i) for i in np.linspace(0.2, 1, N)]

pv.set_plot_theme("document")
cpos = helper.get_camera_position()
brain_cloud = helper.plot_3d_brain()

snr_threshold = 0.5
max_nr_components = 10

for i_sub, (participant, exp) in enumerate(zip(participants, experiments)):
    dfs = []

    participant_id = "%s_%s" % (participant, exp)

    plot_filename = "%s/localization_%s_spatial_max.png" % (
        plot_folder,
        participant_id,
    )

    raw_file = "../working/%s_raw.fif" % participant_id
    raw = mne.io.read_raw_fif(raw_file)
    raw.crop(0, 1)
    raw.load_data()
    raw.pick_types(ecog=True)

    electrodes = helper.plot_electrodes(raw)

    # create mesh plot
    plotter = pv.Plotter(off_screen=True)
    actor = plotter.add_mesh(brain_cloud, color="w")
    act_electrodes = plotter.add_mesh(electrodes, point_size=25, color="w")

    # get individual peaks
    peaks = helper.get_participant_peaks(participant, exp)

    for peak in peaks:

        # load spectral parametrization of SSD traces
        df_file = "%s/sources_%s_peak_%.2f.csv" % (
            results_folder,
            participant_id,
            peak,
        )
        df = pd.read_csv(df_file)
        df = df[df.snr > snr_threshold]

        nr_components = np.min([max_nr_components, len(df)])

        for i_comp in range(nr_components):
            location = df.iloc[i_comp][["x", "y", "z"]].to_numpy()
            max_peak = df.iloc[i_comp]["freq"]
            snr = df.iloc[i_comp]["snr"]

            # make circle with color according to frequency
            cyl = pv.PolyData(location)
            idx_color = np.argmin(np.abs(peaks_n - max_peak))
            color = cmap[idx_color]

            actor = plotter.add_mesh(
                cyl,
                point_size=60 * snr,
                color=color,
                render_points_as_spheres=True,
            )

    plotter.show(
        cpos=cpos,
        title=str(cpos),
        interactive_update=False,
        screenshot=plot_filename,
    )
    plotter.screenshot(plot_filename, transparent_background=True)
    plotter.close()
