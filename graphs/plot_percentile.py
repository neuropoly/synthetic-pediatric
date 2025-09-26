import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm
from collections import defaultdict

def read_zscores(csv_file, colname):
    z_dict = {}
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row.get("subject", "").strip()
            age = row.get("age", "").strip()
            val = row.get(colname, None)
            if subj and age and val is not None and val != "":
                try:
                    z_value = float(val)
                    z_dict[(subj, age)] = z_value
                except ValueError:
                    continue
    return z_dict

def aggregate_by_subject(z_dict):
    subj_data = defaultdict(lambda: {"ages": [], "values": []})
    for (subj, age_str), z_value in z_dict.items():
        try:
            age = float(age_str)
        except ValueError:
            continue
        subj_data[subj]["ages"].append(age)
        subj_data[subj]["values"].append(z_value)
    aggregated = {}
    for subj, data in subj_data.items():
        if data["ages"]:
            mean_age = np.mean(data["ages"])
            mean_z = np.mean(data["values"])
            aggregated[subj] = (mean_age, mean_z)
    return aggregated

def plot_percentile(csv_file1, csv_file2, csv_file3, output_name):

    structures = {
        "dgm": "Deep Grey Matter",
        "cgm": "Cortical Grey Matter",
        "wm":  "White Matter",
        "csf": "Ventricles"
    }
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()
    
    for i, (key, title_str) in enumerate(structures.items()):
        colname = f"subject_mean_roll_z_{key}"
        z1 = read_zscores(csv_file1, colname)
        z2 = read_zscores(csv_file2, colname)
        z3 = read_zscores(csv_file3, colname)
    
        agg1 = aggregate_by_subject(z1)
        agg2 = aggregate_by_subject(z2)
        agg3 = aggregate_by_subject(z3)
    
        common_subjects = set(agg1.keys()) & set(agg2.keys()) & set(agg3.keys())
        if not common_subjects:
            raise ValueError("No common subjects found between the CSV files.")
    
        subjects_data = []
        for subj in common_subjects:
            mean_age = agg1[subj][0]
            p1 = norm.cdf(agg1[subj][1]) * 100
            p2 = norm.cdf(agg2[subj][1]) * 100
            p3 = norm.cdf(agg3[subj][1]) * 100
            diff12 = abs(p1 - p2)
            diff13 = abs(p1 - p3)
            subjects_data.append((mean_age, diff12, diff13))
    
        group_diff12 = defaultdict(list)
        group_diff13 = defaultdict(list)
        for mean_age, d12, d13 in subjects_data:
            rounded_age = int(round(mean_age))
            group_diff12[rounded_age].append(d12)
            group_diff13[rounded_age].append(d13)
    
        unique_rounded_ages = sorted(group_diff12.keys())
        mean_diff12 = np.array([np.mean(group_diff12[age]) for age in unique_rounded_ages])
        mean_diff13 = np.array([np.mean(group_diff13[age]) for age in unique_rounded_ages])

        ax = axs[i]
        ax.plot(unique_rounded_ages, mean_diff12, marker='o', linestyle='-', color='purple', label="GT - Reconstructed")
        ax.plot(unique_rounded_ages, mean_diff13, marker='s', linestyle='-', color='orange', label="GT - Synthetic ULF")
        ax.set_xlabel("Age (months)", fontsize=12)
        ax.set_ylabel("Mean Euclidean Difference", fontsize=12)
        ax.set_title(title_str, fontsize=16)
        ax.axvline(24, color="gray", linestyle="--", linewidth=1, label="24 months")
        ax.tick_params(axis='both', labelsize=12)
    
    axs[0].legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    out_png = f"path/to/save/{output_name}_percentile.png"
    plt.savefig(out_png, dpi=300)

if __name__ == "__main__":

    plot_percentile(
        csv_file1='path/to/csv/zscore.csv',
        csv_file2='path/to/csv/zscore_pred.csv',
        csv_file3='path/to/csv/zscore_ulf.csv',
        output_name='Prediction'
    )