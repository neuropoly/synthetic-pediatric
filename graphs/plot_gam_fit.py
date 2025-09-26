import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.sparse
def to_array(self):
    return self.toarray()
scipy.sparse.spmatrix.A = property(to_array)
import pygam

def rolling_mean(X, Y, window_size=150):
   
    Y_smoothed = np.zeros_like(Y)
    half_w = window_size // 2
    for i in range(len(Y)):
        start = max(0, i - half_w)
        end = min(len(Y), i + half_w + 1)
        Y_smoothed[i] = np.mean(Y[start:end])
    return Y_smoothed

def plot_gam_spline(x, y, color):

    x_vals = np.linspace(x.min(), x.max(), 100)
    gam = pygam.LinearGAM(pygam.s(0), n_splines=15, lam=0.8).fit(x, y)
    y_vals_spline = gam.predict(x_vals)
    plt.plot(x_vals, y_vals_spline, color=color)

    return gam

def plot_population(subject_data, tissue='GM', max_age=None, fit='gam', version_name=None):
    """
    Plots the growth curves of the volume of a given tissue type and a fit method.

    Parameters:
    - subject_data: Dictionary containing subject data.
    - tissue: The type of tissue to plot (deep_GM, cortical_GM, WM, CSF).
    - max_age: Maximum age to consider for the plot.
    - fit: The fitting method to use. Possibilities are 'gam', 'rolling_mean', 'smooth rolling_mean', 'polynomial', or 'spline'.

    """
    all_ages_M = []
    all_volumes_M = []
    all_ages_F = []
    all_volumes_F = []

    for subj, sessions in subject_data.items():
        for ses_id, data in sessions.items():
            age = data['age']
            if max_age is not None and age > max_age:
                continue
            if data['sex'] == 'M':
                all_ages_M.append(data['age'])
                all_volumes_M.append(data[tissue])
            elif data['sex'] == 'F':
                all_ages_F.append(data['age'])
                all_volumes_F.append(data[tissue])

    if not all_ages_M and not all_volumes_M and not all_ages_F and not all_volumes_F:
        print("No data to plot.")
        return

    plt.figure(figsize=(8,6))

    x_M = np.array(all_ages_M)
    y_M = np.array(all_volumes_M)
    x_F = np.array(all_ages_F)
    y_F = np.array(all_volumes_F)

    sorted_indices_M = np.argsort(x_M)
    sorted_indices_F = np.argsort(x_F)
    x_M = x_M[sorted_indices_M]
    y_M = y_M[sorted_indices_M]
    x_F = x_F[sorted_indices_F]
    y_F = y_F[sorted_indices_F]
    

    plt.scatter(x_M, y_M, s=10, color='blue', alpha=0.2, label="Male")
    plt.scatter(x_F, y_F, s=10, color='red', alpha=0.2, label="Female")

    if fit == 'gam':
        gam_M = plot_gam_spline(x_M, y_M, color='blue')
        gam_F = plot_gam_spline(x_F, y_F, color='red')

    elif fit == 'rolling_mean':
        Y_rolling_M = rolling_mean(x_M, y_M, window_size=150)
        Y_rolling_F = rolling_mean(x_F, y_F, window_size=150)
        plt.plot(x_M, Y_rolling_M, '-k', color='blue', label="Male")
        plt.plot(x_F, Y_rolling_F, '-k', color='red', label="Female")

    elif fit == 'smooth rolling_mean':
        X_unique_M, uniq_idx_M = np.unique(x_M, return_index=True)
        X_clean_M = X_unique_M
        Y_rolling_clean = Y_rolling_M[uniq_idx_M]

        X_unique_F, uniq_idx_F = np.unique(x_F, return_index=True)
        X_clean_F = X_unique_F
        Y_rolling_clean = Y_rolling_F[uniq_idx_F]

        rolling_spline = scipy.interpolate.UnivariateSpline(X_clean_M, Y_rolling_clean, s=1.0) 
        Y_rolling_spline_M = rolling_spline(X_clean_M)
        plt.plot(X_clean_M, Y_rolling_spline_M, '-c', color='blue', label="Male")

        rolling_spline = scipy.interpolate.UnivariateSpline(X_clean_F, Y_rolling_clean, s=1.0) 
        Y_rolling_spline_F = rolling_spline(X_clean_F)
        plt.plot(X_clean_F, Y_rolling_spline_F, '-c', color='red', label="Female")

    elif fit == 'polynomial':
        coeffs_M = np.polyfit(x_M, y_M, 3)
        x_vals_M = np.linspace(x_M.min(), x_M.max(), 100)
        y_vals_poly_M = np.polyval(coeffs_M, x_vals_M)
        plt.plot(x_vals_M, y_vals_poly_M, 'g-', color='blue', label="Male")

        coeffs_F = np.polyfit(x_F, y_F, 3)
        x_vals_F = np.linspace(x_F.min(), x_F.max(), 100)
        y_vals_poly_F = np.polyval(coeffs_F, x_vals_F)
        plt.plot(x_vals_F, y_vals_poly_F, 'g-', color='red', label="Female")

    elif fit == 'spline':
        X_unique_M, uniq_idx_M = np.unique(x_M, return_index=True)
        x_M = X_unique_M
        y_M = y_M[uniq_idx_M]

        spline = scipy.interpolate.UnivariateSpline(x_M, y_M, s=0)
        y_vals_spline_M = spline(x_vals_M)
        plt.plot(x_vals_M, y_vals_spline_M, '-m', color='blue', label="Male")

        X_unique_F, uniq_idx_F = np.unique(x_F, return_index=True)
        x_F = X_unique_F
        y_F = y_F[uniq_idx_F]

        spline = scipy.interpolate.UnivariateSpline(x_F, y_F, s=0)
        y_vals_spline_F = spline(x_vals_F)
        plt.plot(x_vals_F, y_vals_spline_F, '-m', color='blue', label="Male")

    elif fit == None:
        pass

    plt.legend(loc='lower right')
    plt.subplots_adjust(left=0.15)

    plt.xlabel("Age (months)")
    plt.ylabel(f"{tissue} Volume (mm³)")
    plt.title(f"Changes in {tissue} Volume Across Age")
    plt.savefig(f"path/to/save/graph/{version_name}_{tissue}_{fit}.png")

    if fit == 'gam':
        return gam_M, gam_F
    
def plot_two_gam_splines(subject_data1, subject_data2, tissues=['deep_GM', 'cortical_GM', 'WM', 'ventricle'], max_age=None, version_name=None):
    """
    Plots GAM spline curves for multiple tissues from two dictionaries.
    This graph is used to compare the trends between two datasets.

    Parameters:
    - subject_data1: Dictionary containing the original dataset.
    - subject_data2: Dictionary containing the predicted dataset.
    - tissues: List of tissue types to plot (e.g., ['deep_GM', 'cortical_GM', 'WM', 'ventricle']).
    - max_age: Maximum age to consider for the plot. If None, all ages are used.
    """

    matching_keys = set()
    for subj, sessions in subject_data1.items():
        for ses_id in sessions.keys():
            if subj in subject_data2 and ses_id in subject_data2[subj]:
                matching_keys.add((subj, ses_id))

    def gather_data(subject_dict, tissue):
        ages_M, volumes_M = [], []
        ages_F, volumes_F = [], []
        for (subj, ses_id) in matching_keys:
            if subj not in subject_dict or ses_id not in subject_dict[subj]:
                continue
            data = subject_dict[subj][ses_id]
            age = data['age']
            if max_age is not None and age > max_age:
                continue
            if data['sex'] == 'M':
                ages_M.append(age)
                volumes_M.append(data[tissue])
            elif data['sex'] == 'F':
                ages_F.append(age)
                volumes_F.append(data[tissue])

        return np.array(ages_M), np.array(volumes_M), np.array(ages_F), np.array(volumes_F)

    def plot_gam(ax, x, y, color, label, linestyle):

        sorted_idx = np.argsort(x)
        x = x[sorted_idx]
        y = y[sorted_idx]

        if len(x) == 0:
            return

        x_vals = np.linspace(x.min(), x.max(), 100)
        gam = pygam.LinearGAM(pygam.s(0), n_splines=15, lam=0.8).fit(x, y)
        y_vals_spline = gam.predict(x_vals)
        ax.plot(x_vals, y_vals_spline, color=color, label=label, linestyle=linestyle)

        std_val = np.std(y)
        return x_vals, y_vals_spline, std_val

    tissue_titles = {
        'deep_GM': 'Deep Grey Matter',
        'cortical_GM': 'Cortical Grey Matter',
        'WM': 'White Matter',
        'ventricule': 'Ventricles'
    }

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()

    colors = ['blue', 'red', 'lightblue', 'lightcoral']

    for i, tissue in enumerate(tissues):
        ax = axs[i]
        xM_1, yM_1, xF_1, yF_1 = gather_data(subject_data1, tissue)
        xM_2, yM_2, xF_2, yF_2 = gather_data(subject_data2, tissue)

        x_vals_M1, y_vals_M1, std_M1 = plot_gam(ax, xM_1, yM_1, color=colors[0],
                                                        label="Male (Ground Truth)", linestyle="-")
        if x_vals_M1 is not None:
            ax.fill_between(x_vals_M1, y_vals_M1 - std_M1, y_vals_M1 + std_M1,
                            color=colors[0], alpha=0.2)
            
        x_vals_F1, y_vals_F1, std_F1 = plot_gam(ax, xF_1, yF_1, color=colors[1],
                                                        label="Female (Ground Truth)", linestyle="-")
        if x_vals_F1 is not None:
            ax.fill_between(x_vals_F1, y_vals_F1 - std_F1, y_vals_F1 + std_F1,
                            color=colors[1], alpha=0.2)

        x_vals_M2, y_vals_M2, std_M2 = plot_gam(ax, xM_2, yM_2, color=colors[2],
                                                        label="Male (Reconstruction)", linestyle="-")
        if x_vals_M2 is not None:
           ax.fill_between(x_vals_M2, y_vals_M2 - std_M2, y_vals_M2 + std_M2,
                           color=colors[2], alpha=0.2)
        x_vals_F2, y_vals_F2, std_F2 = plot_gam(ax, xF_2, yF_2, color=colors[3],
                                                        label="Female (Reconstruction)", linestyle="-")
        if x_vals_F2 is not None:
            ax.fill_between(x_vals_F2, y_vals_F2 - std_F2, y_vals_F2 + std_F2,
                           color=colors[3], alpha=0.2)
        
        ax.scatter(xM_1, yM_1, color=colors[0], marker='o', s=20, alpha=0.3)
        ax.scatter(xF_1, yF_1, color=colors[1], marker='o', s=20, alpha=0.3)
        ax.axvline(x=24, color='gray', linestyle='--', linewidth=1, label="24 months")

        ax.set_title(tissue_titles.get(tissue, tissue)) 
        ax.set_xlabel("Age (months)")
        ax.set_ylabel("Volume (mm³)")
        handles, labels = axs[0].get_legend_handles_labels()
        axs[1].legend(handles, labels, loc='lower right', fontsize='small')

    plt.tight_layout()
    plt.savefig(f"path/to/save/graph/{version_name}_gam_comparison.png", dpi=300)

def main(volume=False, gam_comparison=False):

    if volume:
        with open('path/to/dict/dict_pred.pkl', 'rb') as f:
            loaded_subject_data = pickle.load(f)
        plot_population(loaded_subject_data, tissue='GM', max_age=None, fit='gam')
        plot_population(loaded_subject_data, tissue='WM', max_age=None, fit='gam')
        plot_population(loaded_subject_data, tissue='CSF', max_age=None, fit='gam')
   
    if gam_comparison:
        with open('path/to/dict/dict_volume.pkl', 'rb') as f:
            loaded_subject_data = pickle.load(f)
        with open('path/to/dict/dict_pred.pkl', 'rb') as f:
            loaded_subject_data2 = pickle.load(f)
        plot_two_gam_splines(loaded_subject_data, loaded_subject_data2, version_name='pred')


if __name__ == "__main__":
    main(gam_comparison=True)
