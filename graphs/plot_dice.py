import os
import matplotlib.pyplot as plt
import re
import csv

def plot_dice_scores_by_region():
    """
    This function reads 2 dice score CSV files and plots scatter plots for four brain regions to compare them:
    
    Each CSV file is expected to have the columns:
       subject_id, session_id, age_in_months, dice_wm, dice_deepgm, dice_corticalgm, dice_ventricules
    
    Four scatter plots are produced, one for each region: dice_wm, dice_deepgm, dice_corticalgm, dice_ventricules.

    """
    base_path = "path/to/folder"
    group_files = 'path/to/csv1/dice_pred.csv'
    lf_file = 'path/to/csv2/dice_synth.csv'
    
    group_data = {
        "age": [],
        "dice_wm": [],
        "dice_deepgm": [],
        "dice_corticalgm": [],
        "dice_ventricules": []
    }
    lf_data = {
        "age": [],
        "dice_wm": [],
        "dice_deepgm": [],
        "dice_corticalgm": [],
        "dice_ventricules": []
    }

    with open(group_files, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            try:
                match = re.search(r"ses-(\d+)mo", row["session_id"])
                if not match:
                    continue  # skip if pattern not found
                age = float(match.group(1))

                group_data["age"].append(age)
                group_data["dice_wm"].append(float(row["Dice_WM"]))
                group_data["dice_deepgm"].append(float(row["Dice_DeepGM"]))
                group_data["dice_corticalgm"].append(float(row["Dice_CorticalGM"]))
                group_data["dice_ventricules"].append(float(row["Dice_Ventricles"]))
            except ValueError:
                continue

    if not os.path.exists(lf_file):
        print(f"[ERROR] File not found: {lf_file}")
        return

    with open(lf_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                match = re.search(r"ses-(\d+)mo", row["ID"])
                if not match:
                    continue
                age = float(match.group(1))

                lf_data["age"].append(age)
                lf_data["dice_wm"].append(float(row["Dice_WM"]))
                lf_data["dice_deepgm"].append(float(row["Dice_DeepGM"]))
                lf_data["dice_corticalgm"].append(float(row["Dice_CorticalGM"]))
                lf_data["dice_ventricules"].append(float(row["Dice_Ventricles"]))
            except ValueError:
                continue

    regions = ["dice_wm", "dice_deepgm", "dice_corticalgm", "dice_ventricules"]
    region_titles = {
        "dice_wm": "White Matter",
        "dice_deepgm": "Deep Grey Matter",
        "dice_corticalgm": "Cortical Grey Matter",
        "dice_ventricules": "Ventricules"
    }
    all_dice_scores = (
        group_data["dice_wm"] + group_data["dice_deepgm"] + group_data["dice_corticalgm"] + group_data["dice_ventricules"] +
        lf_data["dice_wm"] + lf_data["dice_deepgm"] + lf_data["dice_corticalgm"] + lf_data["dice_ventricules"]
    )
    y_min = min(all_dice_scores) - 0.1
    y_max = max(all_dice_scores) + 0.1

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i, region in enumerate(regions):
        ax = axs[i]
        ax.scatter(group_data["age"], group_data[region], color="purple", s=6, alpha=0.4, label="Reconstruted 3T")
        ax.scatter(lf_data["age"], lf_data[region], color="orange", alpha=0.4, s=6, label="Synthetic LF")
        ax.set_xlabel("Age (months)", fontsize=14)
        ax.set_ylabel('Dice scores', fontsize=14)
        ax.set_title(region_titles[region], fontsize=16)
        ax.tick_params(axis='both', labelsize=14) 
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=14)
    
    plt.tight_layout()
    output_file = os.path.join(base_path, "dice_scores.png")
    plt.savefig(output_file, dpi=300)
