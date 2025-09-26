import numpy as np
import matplotlib.pyplot as plt
import pygam
import pickle

def plot_subcortical_ratios(subject_data, version_name=None):
    """
    Creates a graph showing the fitted data for the following structures:
    thalamus, caudate, putamen, pallidum, hippocampus, amygdala, cerebellum.
    """

    structures = ['thalamus','caudate','putamen','pallidum','hippocampus','amygdala']

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs = axs.ravel()

    def gather_ratio_data(sd, key_name):
        xM, rM = [], []
        xF, rF = [], []
        for subj, sessions in sd.items():
            for ses_id, data in sessions.items():
                age = data.get('age', None)
                if age is None:
                    continue
                total_vol = data.get('total', None)
                struct_vol = data.get(key_name, None)
                if not total_vol or total_vol == 0:
                    continue
                ratio = struct_vol / total_vol

                if data.get('sex','') == 'M':
                    xM.append(age)
                    rM.append(ratio)
                elif data.get('sex','') == 'F':
                    xF.append(age)
                    rF.append(ratio)
        return np.array(xM), np.array(rM), np.array(xF), np.array(rF)

    def plot_gam(ax, x, y, color, label):
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]

        # Scatter
        ax.scatter(x_sorted, y_sorted, color=color, alpha=0.3, s=10)

        if len(x_sorted) > 1:
            x_vals = np.linspace(x_sorted.min(), x_sorted.max(), 100)
            gam = pygam.LinearGAM(pygam.s(0), n_splines=15, lam=0.8).fit(x_sorted, y_sorted)
            y_vals = gam.predict(x_vals)
            ax.plot(x_vals, y_vals, color=color, label=label)

    for i, structure in enumerate(structures):
        ax = axs[i]
        xM_str, rM_str, xF_str, rF_str = gather_ratio_data(subject_data, structure)

        plot_gam(ax, xM_str, rM_str, 'blue', f"Male ({structure})")
        plot_gam(ax, xF_str, rF_str, 'red', f"Female ({structure})")

        ax.axvline(x=24, color='gray', linestyle='--', linewidth=1, label="24 months")
        ax.set_title(f"{structure.capitalize()} ratio")
        ax.set_xlabel("Age (months)")
        ax.set_ylabel("Ratio of total volume")
        ax.legend(loc='lower right', fontsize='small')

    if len(structures) < 8:
        axs[-1].set_visible(False)

    plt.tight_layout()

    out_plot = f"path/to/save/{version_name}_subcortical_ratios.png"
    plt.savefig(out_plot)

def plot_three_dicts_subcortical_ratios(subject_data1, subject_data2, subject_data3, version_name=None):
    """
    This fucntion plots the subcortical ratios from three different dictionaries on the same graph.

    The 7 subcortical structures are: 
      ['thalamus','caudate','putamen','pallidum','hippocampus','amygdala','cerebellum'].
      
    Parameters:
      subject_data1, subject_data2, subject_data3: dictionaries with keys as subject IDs, and values as dicts of session data.

    """

    structures = ['thalamus','caudate','putamen','pallidum','hippocampus','amygdala']

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs = axs.ravel()

    def gather_ratio_data(sd, key_name):
        ages = []
        ratios = []
        for subj, sessions in sd.items():
            for ses_id, data in sessions.items():
                age = data.get('age', None)
                if age is None:
                    continue
                total_vol = data.get('total', None)
                struct_vol = data.get(key_name, None)
                if not total_vol or total_vol == 0:
                    continue
                ratio = struct_vol / total_vol
                ages.append(age)
                ratios.append(ratio)
        return np.array(ages), np.array(ratios)

    def plot_gam_fit(ax, x, y, color, label):
        if len(x) == 0:
            return

        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]

        if len(x_sorted) > 1:
            x_vals = np.linspace(x_sorted.min(), x_sorted.max(), 100)
            gam = pygam.LinearGAM(pygam.s(0), n_splines=15, lam=0.8).fit(x_sorted, y_sorted)
            y_vals = gam.predict(x_vals)

            std_val = np.std(y_sorted)
            ax.plot(x_vals, y_vals, color=color, label=label)
            ax.fill_between(x_vals, y_vals - std_val, y_vals + std_val, color=color, alpha=0.2)

    for i, structure in enumerate(structures):
        ax = axs[i]

        x1, y1 = gather_ratio_data(subject_data1, structure)
        x2, y2 = gather_ratio_data(subject_data2, structure)
        x3, y3 = gather_ratio_data(subject_data3, structure)
        
        plot_gam_fit(ax, x1, y1, color='green', label="GT")
        plot_gam_fit(ax, x2, y2, color='orange', label="Synth ULF")
        plot_gam_fit(ax, x3, y3, color='purple', label={version_name})
        
        ax.axvline(x=24, color='gray', linestyle='--', linewidth=1, label="24 months")
        ax.set_title(f"{structure.capitalize()} ratio", fontsize=18)
        ax.set_xlabel("Age (months)", fontsize=16)
        ax.set_ylabel("Fraction of total volume", fontsize=16)
        ax.tick_params(axis='both', labelsize=16) 
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(handles, labels, loc='upper right', fontsize='16')

    if len(structures) < len(axs):
        axs[-1].set_visible(False)

    plt.tight_layout()
    out_plot = f"path/to/save/{version_name}_subcortical_ratios.png"
    plt.savefig(out_plot, dpi=300)

    if __name__ == "__main__":
        with open('path/to/dict/dict_volume.pkl', 'rb') as f:
            loaded_subject_data = pickle.load(f)
        with open('path/to/dict/dict_ulf.pkl', 'rb') as f:
            loaded_subject_data2 = pickle.load(f)
        with open('path/to/dict/dict_pred.pkl', 'rb') as f:
            loaded_subject_data3 = pickle.load(f)
        plot_three_dicts_subcortical_ratios(loaded_subject_data, loaded_subject_data2, loaded_subject_data3, version_name='Reconstructed')