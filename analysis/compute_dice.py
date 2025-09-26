import os
import re
import csv
import nibabel as nib
import numpy as np
import re


WM_LABELS = [2, 7, 13, 16, 28, 41, 46, 52, 60]
DEEPGM_LABELS = [10, 49, 11, 50, 12, 51, 13, 52]
CORTICALGM_LABELS = [3, 17, 53, 18, 54, 8, 42, 47]
VENTRICULES_LABELS = [4, 5, 14, 15, 43, 44]
TOTAL_LABELS = WM_LABELS + DEEPGM_LABELS + CORTICALGM_LABELS + VENTRICULES_LABELS
THALAMUS_LABELS = [10, 49]
CAUDATE_LABELS = [11, 50]
PUTAMEN_LABELS = [12, 51]
PALLIDUM_LABELS = [13, 52]
HIPPOCAMPUS_LABELS = [17, 53]
AMYGDALA_LABELS = [18, 54]
CEREBELLUM_LABELS = [7, 8, 46, 47]

LABEL_GROUPS = {
    "WM": WM_LABELS,
    "DeepGM": DEEPGM_LABELS,
    "CorticalGM": CORTICALGM_LABELS,
    "Ventricles": VENTRICULES_LABELS,
    "Thalamus": THALAMUS_LABELS,
    "Caudate": CAUDATE_LABELS,
    "Putamen": PUTAMEN_LABELS,
    "Pallidum": PALLIDUM_LABELS,
    "Hippocampus": HIPPOCAMPUS_LABELS,
    "Amygdala": AMYGDALA_LABELS,
    "Cerebellum": CEREBELLUM_LABELS
}

def compute_dice(mask1, mask2, label):
    bin1 = (mask1 == label).astype(np.uint8)
    bin2 = (mask2 == label).astype(np.uint8)
    inter = np.sum(bin1 * bin2)
    size1 = np.sum(bin1)
    size2 = np.sum(bin2)
    if size1 + size2 == 0:
        return 1.0 
    return 2 * inter / (size1 + size2)

def extract_id(filename):
    match = re.match(r"(sub-\d+_ses-[^_]+)", filename)
    return match.group(1) if match else None

def main(pred_dirs, gt_dir, output_csv):
    results = []

    for pred_dir in pred_dirs:
        pred_files = [
            f for f in os.listdir(pred_dir)
            if not f.startswith("._") and not f.startswith(".") and (f.endswith(".nii") or f.endswith(".nii.gz"))
        ]

    gt_files = [
    f for f in os.listdir(gt_dir)
    if not f.startswith("._") and not f.startswith(".") and (f.endswith(".nii") or f.endswith(".nii.gz"))]

    for pred_file in pred_files:
        pred_id = extract_id(pred_file)
        if not pred_id:
            continue
        gt_match = next((f for f in gt_files if pred_id in f), None)
        if not gt_match:
            print(f"Ground truth not found for {pred_id}")
            continue

        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_match)

        pred_data = np.round(nib.load(pred_path).get_fdata())
        gt_data = np.round(nib.load(gt_path).get_fdata())

        row = {"ID": pred_id}
        for group_name, labels in LABEL_GROUPS.items():
            dice_vals = [compute_dice(pred_data, gt_data, label) for label in labels]
            row[f"Dice_{group_name}"] = np.mean(dice_vals)

        results.append(row)

    # Write to CSV
    fieldnames = ["ID"] + [f"Dice_{k}" for k in LABEL_GROUPS.keys()]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {output_csv}")
