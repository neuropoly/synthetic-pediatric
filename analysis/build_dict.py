import os
import re
import glob
import numpy as np
import nibabel as nib

def load_exclusions_from_yaml(yaml_path):
    """
    Parse exclude.yaml and return two sets:
     - t1w_excludes = set of "sub-XXXXXX_ses-YYY"
     - t2w_excludes = set of "sub-XXXXXX_ses-YYY"
    """
    with open(yaml_path, 'r') as f:
        lines = f.read().splitlines()

    t1w_excludes = set()
    t2w_excludes = set()
    mode_t1 = False
    mode_t2 = False

    for line in lines:
        line = line.strip()
        # Skip blank lines
        if not line:
            continue
        if line.startswith('#'):
            # Switch mode based on comment
            if 'T1w' in line:
                mode_t1 = True
                mode_t2 = False
            elif 'T2w' in line:
                mode_t2 = True
                mode_t1 = False
            continue
        
        entry = re.split(r'#', line, maxsplit=1)[0].lstrip('-').strip()
        match = re.search(r'(sub-\d+_ses-\d+(?:mo|wk))', entry)
        if match:
            actual_key = match.group(1)
            if mode_t1:
                t1w_excludes.add(actual_key)
            elif mode_t2:
                t2w_excludes.add(actual_key)

    return t1w_excludes, t2w_excludes

def dict_volume_regions(path_dataset):
    """
    Creates a nested dictionary to store the subject ID, the age at the session, and
    the WM, deep_GM, cortical_GM, CSF (only ventricules), total volumes and separate regions volumes for thalamus, 
    caudate, putamen, pallidum, hippocampus and amygdala from segmentation files in a BIDS dataset.

    Parameters
    ----------
    path_dataset : str
        Path to the segmentation folder of a BIDS dataset directory.

    Returns
    -------
    subject_data : dict
        Nested dict of the form:
        {
          "sub-XXXXXX": {
            "ses-YYY": {
              "age": float,
              "WM": float,
              etc. for each region
              "sex": str (M or F)
            },
            ...
          },
          ...
        }
    """
    # Load excludes from YAML file
    yaml_path = 'exclude.yaml'
    yaml_seg_path = 'exclude_seg.yaml'
    t1w_excluded_im, t2w_excluded_im = load_exclusions_from_yaml(yaml_path)
    t1w_excluded_seg, t2_excluded_seg = load_exclusions_from_yaml(yaml_seg_path)
    
    # Read the sex of the participants from the participants.tsv file
    subject_sex_map = {}
    with open('participants.tsv', 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:
        cols = line.strip().split('\t') 
        if len(cols) < 2:
            continue
        subject_id_tsv = cols[0]
        session_id_tsv = cols[2].split(',')
        sex = cols[1]
        
        if subject_id_tsv not in subject_sex_map:
            subject_sex_map[subject_id_tsv] = {}
        
        for session_id_tsv in session_id_tsv:
            subject_sex_map[subject_id_tsv][session_id_tsv] = sex
    
    subject_data = {}

    seg_files = glob.glob(os.path.join(path_dataset, "sub-*", "ses-*", "anat", f"*T1w_seg.nii.gz"))

    print(f"Found {len(seg_files)} segmentation files.")
    
    for seg_file in seg_files:
        match_sub = re.search(r"(sub-\d+)_ses-\d+(?:mo|wk)", os.path.basename(seg_file))
        if not match_sub:
            print(f"Skipping {seg_file}: no sub-xxx_ses-xxx found.")
            continue
        subject_session = match_sub.group(0)
        parts = subject_session.split('_', 1)
        subject_id = parts[0]
        session_id = parts[1]

        # Check if the file is in the exclude list
        combined_key = f"{subject_id}_{session_id}"
        if combined_key in t1w_excluded_im or combined_key in t1w_excluded_seg:
            print(f"Excluding {combined_key} from T1w")
            continue

        # Extract numeric age
        match_age = re.search(r"ses-(\d+)(mo|wk)", session_id)
        if not match_age:
            continue
        age_value = int(match_age.group(1))
        age_unit = match_age.group(2)
        if age_unit == "mo":
            age = float(age_value)
        else:  # 'wk' → convert weeks to months
            age = float(age_value) / 4.0

        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata()
        seg_data = np.round(seg_data).astype(int)
        voxel_size = seg_img.header.get_zooms()

        # Compute volumes
        label_volume_WM = np.sum(np.isin(seg_data, [2,7,13,16,28,41,46,52,60])) * np.prod(voxel_size)
        label_volume_deepGM = np.sum(np.isin(seg_data, [10,49,11,50,12,51,13,52])) * np.prod(voxel_size)
        label_volume_corticalGM = np.sum(np.isin(seg_data, [3,17,53,18,54,8,42,47])) * np.prod(voxel_size)
        label_volume_ventricule = np.sum(np.isin(seg_data, [4,5,14,15,43,44])) * np.prod(voxel_size) # Without CSF
        label_volume_total = np.sum(np.isin(seg_data, [3,8,10,11,12,17,18,26,42,47,49,50,51,53,54,58,2,7,13,16,28,41,46,52,60,4,5,14,15,43,44])) * np.prod(voxel_size)
        label_volume_thalamus = np.sum(np.isin(seg_data, [10,49])) * np.prod(voxel_size)
        label_volume_caudate = np.sum(np.isin(seg_data, [11,50])) * np.prod(voxel_size)
        label_volume_putamen = np.sum(np.isin(seg_data, [12,51])) * np.prod(voxel_size)
        label_volume_pallidum = np.sum(np.isin(seg_data, [13,52])) * np.prod(voxel_size)
        label_volume_hippocampus = np.sum(np.isin(seg_data, [17,53])) * np.prod(voxel_size)
        label_volume_amygdala = np.sum(np.isin(seg_data, [18,54])) * np.prod(voxel_size)

        # Initialize sub-dict
        if subject_id not in subject_data:
            subject_data[subject_id] = {}
        if session_id not in subject_data[subject_id]:
            subject_data[subject_id][session_id] = {}

        subject_data[subject_id][session_id]['age'] = age
        subject_data[subject_id][session_id]['WM'] = label_volume_WM
        subject_data[subject_id][session_id]['deep_GM'] = label_volume_deepGM
        subject_data[subject_id][session_id]['cortical_GM'] = label_volume_corticalGM   
        subject_data[subject_id][session_id]['ventricule'] = label_volume_ventricule
        subject_data[subject_id][session_id]['total'] = label_volume_total
        subject_data[subject_id][session_id]['thalamus'] = label_volume_thalamus
        subject_data[subject_id][session_id]['caudate'] = label_volume_caudate
        subject_data[subject_id][session_id]['putamen'] = label_volume_putamen
        subject_data[subject_id][session_id]['pallidum'] = label_volume_pallidum
        subject_data[subject_id][session_id]['hippocampus'] = label_volume_hippocampus
        subject_data[subject_id][session_id]['amygdala'] = label_volume_amygdala

        if subject_id in subject_sex_map:
            subject_data[subject_id][session_id]['sex'] = subject_sex_map[subject_id][session_id]
        else:
            subject_data[subject_id][session_id]['sex'] = 'Unknown'

    return subject_data
