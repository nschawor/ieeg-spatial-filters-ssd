""" Plots the asymmetry index on a 3d brain.
"""
import numpy as np
import matplotlib.pyplot as plt
import helper
import pandas as pd
import os
import pyvista as pv

df = pd.read_csv("../csv/selected_datasets.csv", index_col=False)
df = df[df.is_rest]

participants = df.participant
experiments = df.experiment
pv.set_plot_theme("document")
nr_components = 10
nr_seconds = 5

results_folder = "../results/bursts/"
results_folder2 = "../results/spec_param/"
plot_folder = "../results/3dbrains/"

os.makedirs(results_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

peaks_n = np.arange(5, 21, 1)
peaks_n = np.arange(0, 0.21, 0.01)
N = len(peaks_n)
cmap = [plt.cm.plasma(i) for i in np.linspace(0.2, 1, N)]

cpos = helper.get_camera_position()
cloud = helper.plot_3d_brain()


# -- create 3d brain plot
plotter = pv.Plotter(off_screen=True)
actor = plotter.add_mesh(cloud)

for i_sub, (participant, exp) in enumerate(zip(participants, experiments)):

    participant_id = "%s_%s" % (participant, exp)

    plot_filename = "%s/asymmetry_%s_spatial_max.png" % (
        plot_folder,
        participant_id,
    )

    # get individual peaks
    peaks = helper.get_participant_peaks(participant, exp)

    for peak1 in peaks:

        df_file_name = "%s/bursts_%s_peak_%s.csv" % (
            results_folder,
            participant_id,
            peak1,
        )

        df = pd.read_csv(df_file_name)
        df = df.set_index("feature")
        df = df.T
        frequency = 1000 / df.period
        if np.any(frequency < 5):
            continue

        asymmetry = 2 * np.abs(df["time_ptsym"] - 0.5)
        asymmetry = asymmetry.to_numpy()

        peak1 = float(peak1)
        df_file = "%s/sources_%s_peak_%.2f.csv" % (
            results_folder2,
            participant_id,
            peak1,
        )
        df = pd.read_csv(df_file)
        df = df[df.snr > 0.5]

        nr_components = np.min([1, len(df)])

        for i_comp in range(nr_components):
            location = df.iloc[i_comp][["x", "y", "z"]].to_numpy()
            location[0] += 2
            max_peak = df.iloc[i_comp]["freq"]
            snr = df.iloc[i_comp]["snr"]

            # make circle with color according to frequency
            cyl = pv.PolyData(location)
            idx_color = np.argmin(np.abs(peaks_n - asymmetry))

            color = cmap[idx_color]

            actor = plotter.add_mesh(
                cyl,
                point_size=snr * 30,
                color=color,
                render_points_as_spheres=True,
            )

plot_filename = "../figures/3dbrain_asymmetry.png"

plotter.show(
    cpos=cpos,
    title=str(cpos),
    interactive_update=False,
    screenshot=plot_filename,
)

plotter.screenshot(plot_filename, transparent_background=True)
plotter.close()
