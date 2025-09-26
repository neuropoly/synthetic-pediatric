import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_zscore_differences():
    """
    This function reads two CSV files containing z-scores and plots histograms of the differences
    in z-scores for four tissue types: deep GM, cortical GM, WM, and CSF.
    """
    csv1_path = "path/to/csv/zscore1.csv"
    csv2_path = "path/to/csv/zscore2.csv"
    
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    df1.columns = df1.columns.str.strip().str.lower()
    df2.columns = df2.columns.str.strip().str.lower()
    
    df1['subject'] = df1['subject'].astype(str).str.strip()
    df1['age'] = df1['age'].astype(str).str.strip()
    df2['subject'] = df2['subject'].astype(str).str.strip()
    df2['age'] = df2['age'].astype(str).str.strip()
    
    df1 = df1.sort_values(by=['subject', 'age']).reset_index(drop=True)
    df2 = df2.sort_values(by=['subject', 'age']).reset_index(drop=True)
    
    if not (df1[['subject', 'age']].equals(df2[['subject', 'age']])):
        raise ValueError("The subject/age rows do not match between the two CSV files.")

    measures = ["zscore_dgm_rolling", "zscore_cgm_rolling", "zscore_wm_rolling", "zscore_csf_rolling"]
    subj_mean_keys = {
        "zscore_dgm_rolling": "subject_mean_roll_z_dgm",
        "zscore_cgm_rolling": "subject_mean_roll_z_cgm",
        "zscore_wm_rolling": "subject_mean_roll_z_wm",
        "zscore_csf_rolling": "subject_mean_roll_z_csf",
    }

    title_map = {
    "zscore_dgm_rolling": "Deep Grey Matter",
    "zscore_cgm_rolling": "Cortical Grey Matter",
    "zscore_wm_rolling":  "White Matter",
    "zscore_csf_rolling": "Ventricles"
    }

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.ravel()
    
    try:
        age_vals = df1['age'].astype(float)
    except Exception as e:
        raise ValueError("Age column could not be converted to numeric.") from e
    
    results_list = []
    # Loop over each measure.
    for i, meas in enumerate(measures):
        if meas not in df1.columns or meas not in df2.columns:
            raise KeyError(f"Expected column '{meas}' not found in one of the CSV files.")
        
        subj_mean_key = subj_mean_keys[meas]
        if subj_mean_key not in df1.columns:
            raise KeyError(f"Expected subject mean column '{subj_mean_key}' not found in CSV1.")
        
        dev_csv1 = np.abs(df1[meas] - df1[subj_mean_key])
        dev_csv2 = np.abs(df2[meas] - df1[subj_mean_key])

        result = dev_csv1 - dev_csv2
        result = result[result >= -1.5]
        results_list.append(result)
        mean_result = np.mean(result)
        std_result = np.std(result)
        
        ax = axs[i]
        ax.hist(result, bins=30, color="purple", edgecolor='black', alpha=0.7)
        ax.set_xlabel("Difference in z-scores", fontsize=14)
        ax.set_ylabel("Number of subjects", fontsize=14)
        ax.set_title(title_map.get(meas, meas), fontsize=16)
        ax.axvline(mean_result, color="black", linestyle=":", label=f"Mean = {mean_result:.2f}")
        ax.axvline(mean_result + std_result, color="green", linestyle="--", label=f"std = {std_result:.2f}")
        ax.axvline(mean_result - std_result, color="green", linestyle="--")
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(loc="upper left", fontsize=14)

        global_xmin = min(r.min() for r in results_list)
        global_xmax = max(r.max() for r in results_list)

        num_bins = 30
        bin_edges = np.linspace(global_xmin, global_xmax, num_bins + 1)
    
        max_count = 0
        for r in results_list:
            counts, _ = np.histogram(r, bins=bin_edges)
            if counts.max() > max_count:
                max_count = counts.max()
    
        for ax in axs:
            ax.set_xlim(global_xmin, global_xmax)
            ax.set_ylim(0, 175)
    
    plt.tight_layout()
    plt.savefig("path/to/save/zscore_diff.png", dpi=300)




    