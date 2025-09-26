import os
import glob
import re
import numpy as np
import nibabel as nib
from synthesis import create_synthetic_image
import matplotlib.pyplot as plt


# Creating the ratios to degrade the images from 3T to 64mT
constants = {'CSF':                {'C': 4.322, 'b': -0.006,'r': 1.0}, 
            'Brainstem':           {'C': 0.459, 'b': 0.508, 'r': 1.0},
            'Ventricules':         {'C': 4.322, 'b': -0.006,'r': 1.0},
            'WM':                  {'C': 0.583, 'b': 0.376, 'r': 1.0}, 
            'Putamen':             {'C': 0.855, 'b': 0.352, 'r': 1.0}, 
            'Cerebral Cortex':     {'C': 0.857, 'b': 0.376, 'r': 1.0},
            'Cerebellum WM':       {'C': 0.583, 'b': 0.376, 'r': 1.0},
            'Cerebellum cortex':   {'C': 0.857, 'b': 0.376, 'r': 1.0}, 
            'Thalamus':            {'C': 0.817, 'b': 0.357, 'r': 1.0},
            'Caudate':             {'C': 0.954, 'b': 0.325, 'r': 1.0}, 
            'Pallidum':            {'C': 0.664, 'b': 0.367, 'r': 1.0},
            'Hippocampus':         {'C': 0.817, 'b': 0.357, 'r': 1.1847}, 
            'Amygdala':            {'C': 0.817, 'b': 0.357, 'r': 1.2475}, 
            'Accumbens Area':      {'C': 0.817, 'b': 0.357, 'r': 1.1395}, 
            'Ventral DC':          {'C': 0.817, 'b': 0.357, 'r': 0.9456},}

def compute_ratios(constants):
    """
    Returns a NumPy array of length 16.
    - The first element (background) is 1.0
    - The next 15 elements are ratio = T_low / T_high 
        for each region in the order they appear in 'constants'.
    """
    B_low = 0.064
    B_high = 3.0
    ratio_array = [1.0]

    for region, params in constants.items():
        c = params['C']
        b = params['b']
        r = params['r']
        T_low = c * (B_low**b)
        T_high = c * (B_high**b)
        ratio = (T_low / T_high) * r
        ratio_array.append(ratio)

    return np.array(ratio_array, dtype=np.float32)
ratios = compute_ratios(constants)

def compute_mean_stds(stds_base_path):
    """
    Computes the mean of five stds arrays that each contain the stds for each regions.
    """
    subfolders = [
        f for f in os.listdir(stds_base_path)
        if os.path.isdir(os.path.join(stds_base_path, f))
    ]

    stds_list = []
    for subfolder in subfolders:
        stds_file = os.path.join(stds_base_path, subfolder, "stds.npy")
        if os.path.isfile(stds_file):
            stds_array = np.load(stds_file)
            stds_list.append(stds_array)

    if len(stds_list) < 5:
        print("Less than five STD arrays found. Returning None.")
        return None

    # Compute element-wise mean of the five entries
    stacked_stds = np.stack(stds_list[:5], axis=0)
    mean_stds = np.sqrt(np.mean(np.square(stacked_stds), axis=0))
    mean_stds /= 1000 #Convert from to ms to seconds
    return mean_stds

stds = compute_mean_stds('/home/ge.polymtl.ca/almahj/ds004611/T1_values/')

def create_synthetic_dataset(bids_root, ratios, stds):
    """
    Creates synthetic images for each subject/session, starting at session-3mo.
    The images are saved inside the synthetic derivatives folder of the same dataset.
    
    Args:
        bids_root (str): Path to the BIDS root folder.
    """

    seg_base = os.path.join(bids_root, "derivatives", "segmentation")
    out_base = os.path.join(bids_root, "derivatives", "synthetic")
    os.makedirs(out_base, exist_ok=True)

    subject_paths = glob.glob(os.path.join(seg_base, "sub-*"))
    for sub_path in subject_paths:
        sub_id = os.path.basename(sub_path)
        session_paths = glob.glob(os.path.join(sub_path, "ses-*"))
        for ses_path in session_paths:
            ses_id = os.path.basename(ses_path)
            
            # Extract the month value (e.g. 15 from "ses-15mo")
            match = re.search(r"ses-(\d+)mo", ses_id)
            if not match:
                continue
            month_num = int(match.group(1))
            if month_num < 3:
                continue

            mod = "T1w"
            anat_folder = os.path.join(ses_path, "anat")
            seg_pattern = f"{sub_id}_{ses_id}_run-*_{{mod}}_seg.nii.gz".replace("{mod}", mod)
            seg_candidates = glob.glob(os.path.join(anat_folder, seg_pattern))
            if not seg_candidates:
                continue
            segmentation_path = seg_candidates[0]
            
            # Load the matching means array for this session, created in the t1_values.py script
            means_file = f"region_array_{month_num}.npy"
            means_path = os.path.join("T1_values/", means_file)
            if not os.path.exists(means_path):
                continue
            means = np.load(means_path)
            means = means * ratios
            
            synthetic_folder = os.path.join(out_base, sub_id, ses_id, "anat")
            os.makedirs(synthetic_folder, exist_ok=True)
            
            create_synthetic_image(
                segmentation=segmentation_path,
                means=means,
                stds=stds,
                output_dir=synthetic_folder,
                current_resolution=(1.0, 1.0, 1.0), 
                sigma=1.15,
                resolution=(2.0, 2.0, 2.0)
            )

            print(f"Synthetic image created for {sub_id} {ses_id} at {synthetic_folder}")

def create_synthetic_dataset_variations(images_folder, bids_folder, ratios, stds):
    """
    Creates 5 different synthetic datasets from a directory of segmentation files.

    For each segmentation file, the function uses a mean array and then creates 5 variations:
      1) means + (-1.0 * stds)
      2) means + ( 0.5 * stds)
      3) means + ( 0.0 * stds)
      4) means + ( 0.5 * stds)
      5) means + ( 1.0 * stds)

    Args:
        images_folder (str): Path to the folder that directly contains the segmentation files.
        bids_folder (str): Path to the parent BIDS folder where derivatives will be stored.
        ratios (ndarray): Per-region scaling factors (from compute_ratios).
        stds (ndarray): Per-region standard deviations.
    """

    fractions = [-1.0, 0.5, 0.0, 0.5, 1.0]

    out_base = os.path.join(bids_folder, "derivatives", "synthetic_V9")
    os.makedirs(out_base, exist_ok=True)

    seg_files = glob.glob(os.path.join(images_folder, "*_seg.nii.gz"))
    count_created = 0

    for seg_path in seg_files:
        filename = os.path.basename(seg_path)

        subject_match = re.search(r"(sub-\d{6})", filename)
        if not subject_match:
            print(f"Skipping {filename} (no 'sub-XXXXXX' pattern found).")
            continue
        subject_id = subject_match.group(1)

        session_match = re.search(r"(ses-\d+mo)", filename)
        if not session_match:
            print(f"Skipping {filename} (no 'ses-XXmo' pattern found).")
            continue
        session_id = session_match.group(1)

        month_num_match = re.search(r"ses-(\d+)mo", filename)
        if not month_num_match:
            print(f"Skipping {filename} (no numerical month extracted).")
            continue
        month_num = int(month_num_match.group(1))

        means_file = f"region_array_{month_num}.npy"
        means_path = os.path.join("path/to/arrays", means_file)
        if not os.path.exists(means_path):
            print(f"Skipping {filename} (no region_array_{month_num}.npy found).")
            continue

        base_means = np.load(means_path) * ratios

        # Check if this subject/session folder already exists; if so, skip
        subject_session_folder = os.path.join(out_base, subject_id, session_id)
        if os.path.exists(subject_session_folder):
            print(f"Skipping {subject_id} {session_id}: folder already exists.")
            continue
        
        # Create the five variations of the synthetic image
        for i, frac in enumerate(fractions, start=1):
            means_variation = base_means + (frac * stds)
            
            variation_folder = os.path.join(out_base, subject_id, session_id, "anat")
            os.makedirs(variation_folder, exist_ok=True)
            out_name = f"{subject_id}_{session_id}_synthetic{i}.nii.gz"

            create_synthetic_image(
                segmentation=seg_path,
                means=means_variation,
                stds=stds,
                output_dir=variation_folder,
                current_resolution=(1.0, 1.0, 1.0),
                sigma=1.15,
                resolution=(2.0, 2.0, 2.0),
                output_name=out_name
            )

            count_created += 1
            print(f"[Variation {i}] Synthetic image created: {out_name} -> {variation_folder}")

    print(f"Finished creating {count_created} synthetic images (5 variations per file).")
