import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_intra_trajectories(subject_data, tissue='GM'):
    """
    Calculates slopes between consecutive sessions for each subject using the specified tissue,
    using the dictionary loaded from a pickle file and built using the build_dict.py script.
    Creates a plot with individual subject data points connected by lines.
    """

    plt.figure()
    for subj, sessions in subject_data.items():
        data_points = []
        sex_for_subj = None

        for sess, data in sessions.items():
            if sex_for_subj is None:
                sex_for_subj = data['sex']
            data_points.append((data['age'], data[tissue]))

        data_points.sort(key=lambda x: x[0])
        ages = [dp[0] for dp in data_points]
        volumes = [dp[1] for dp in data_points]

        color = 'blue' if sex_for_subj == 'M' else 'red'

        # Plot each subject line in the chosen color
        plt.plot(ages, volumes, marker='o', markersize=3, alpha=0.2, color=color, label=subj)
    
    plt.ylabel(f"{tissue} Volume (mm³)")
    plt.title(f"Individual {tissue} Volumes Across Age")
    plt.subplots_adjust(left=0.15)
    plt.savefig(f"path/to/save/graph/trajectories_{tissue}.png")

with open('path/to/dict/dict_volume.pkl', 'rb') as f:
    loaded_subject_data = pickle.load(f)
plot_intra_trajectories(loaded_subject_data, tissue='CSF')
plot_intra_trajectories(loaded_subject_data, tissue='WM')
plot_intra_trajectories(loaded_subject_data, tissue='GM')