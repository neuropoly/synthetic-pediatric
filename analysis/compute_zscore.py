import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from graphs.plot_gam_fit import plot_population
import pickle
import pandas as pd
import csv

def compute_zscore(
    subject_data,
    gam_M_dGM, gam_F_dGM,
    gam_M_cGM, gam_F_cGM,
    gam_M_WM, gam_F_WM,
    gam_M_CSF, gam_F_CSF,
    version_name,
    max_age=None,
    rolling_window=15, 
    csv_file=True,
):
    """
    This function writes a csv file with z-scores for each session in the input dictionary.
    A standard deviation is computed per age for each tissue type (deep GM, cortical GM, WM, CSF).
    The standard deviation is smoothed using a rolling mean with the specified window size.
    The z-score is computed in two ways: using the normal stdev and using the rolling-mean stdev.
    The CSV file contains the following columns:
         subject, age, sex,
         zscore_dgm, zscore_cgm, zscore_wm, zscore_csf, mean_zscore,
         zscore_dgm_rolling, zscore_cgm_rolling, zscore_wm_rolling, zscore_csf_rolling, mean_zscore_rolling,
         subject_mean_roll_z, difference_z, n_sessions
    """

    from collections import defaultdict
    
    age_groups_dGM = defaultdict(list)  
    age_groups_cGM = defaultdict(list)
    age_groups_WM = defaultdict(list)  
    age_groups_CSF = defaultdict(list)  

    all_points = defaultdict(lambda: {"dgm": None, "cgm": None, "wm": None, "csf": None, "sex": None})

    for subj, sessions in subject_data.items():
        for ses_id, data in sessions.items():
            age = data["age"]
            if max_age is not None and age > max_age:
                continue
            sex = data["sex"]
            dgm_val = data.get("deep_GM", None)
            cgm_val = data.get("cortical_GM", None)
            wm_val = data.get("WM", None)
            csf_val = data.get("CSF", None)

            if dgm_val is not None:
                age_groups_dGM[age].append(dgm_val)
            if cgm_val is not None:
                age_groups_cGM[age].append(cgm_val)
            if wm_val is not None:
                age_groups_WM[age].append(wm_val)
            if csf_val is not None:
                age_groups_CSF[age].append(csf_val)

            key = (subj, age)
            all_points[key]["sex"] = sex
            if dgm_val is not None:
                all_points[key]["deep_gm"] = dgm_val
            if cgm_val is not None:
                all_points[key]["cortical_gm"] = cgm_val
            if wm_val is not None:
                all_points[key]["wm"] = wm_val
            if csf_val is not None:
                all_points[key]["csf"] = csf_val

    def compute_std_dict(age_groups):
        """Return a dict {age: stdev} and a sorted list of nonzero (age, stdev)."""
        age_std = {}
        for age, volumes in age_groups.items():
            if len(volumes) > 1:
                age_std[age] = np.std(volumes, ddof=1)
            else:
                age_std[age] = 0.0
        nonzero = [(a, s) for a, s in age_std.items() if s != 0]
        nonzero.sort(key=lambda x: x[0])
        return age_std, nonzero

    dgm_std, dgm_nonzero = compute_std_dict(age_groups_dGM)
    cgm_std, cgm_nonzero = compute_std_dict(age_groups_cGM)
    wm_std, wm_nonzero = compute_std_dict(age_groups_WM)
    csf_std, csf_nonzero = compute_std_dict(age_groups_CSF)

    def make_rolling_std_lookup(nonzero_list):
        if not nonzero_list:
            return lambda age: 0.0
        ages_nonzero = [n[0] for n in nonzero_list]
        stds_nonzero = [n[1] for n in nonzero_list]
        rolling_series = pd.Series(stds_nonzero).rolling(
            window=rolling_window, center=True, min_periods=1
        ).mean()
        roll_pairs = list(zip(ages_nonzero, rolling_series))

        def get_roll_std(age):
            if not roll_pairs:
                return 0.0
            best_age = ages_nonzero[0]
            best_std = roll_pairs[0][1]
            best_dist = abs(age - best_age)
            for (a, s) in roll_pairs:
                dist = abs(age - a)
                if dist < best_dist:
                    best_dist = dist
                    best_age = a
                    best_std = s
            return best_std if not pd.isna(best_std) else 0.0

        return get_roll_std

    get_roll_dgm = make_rolling_std_lookup(dgm_nonzero)
    get_roll_cgm = make_rolling_std_lookup(cgm_nonzero)
    get_roll_wm = make_rolling_std_lookup(wm_nonzero)
    get_roll_csf = make_rolling_std_lookup(csf_nonzero)


    # Compute z-scores per row
    rows_raw = []

    sorted_keys = sorted(all_points.keys(), key=lambda x: (x[0], x[1]))
    for (subj, age) in sorted_keys:
        dgm_vol = all_points[(subj, age)]["deep_gm"]
        cgm_vol = all_points[(subj, age)]["cortical_gm"]
        wm_vol = all_points[(subj, age)]["wm"]
        csf_vol = all_points[(subj, age)]["csf"]
        sex = all_points[(subj, age)]["sex"]

        dgm_sd = dgm_std.get(age, 0)
        cgm_sd = cgm_std.get(age, 0)
        wm_sd = wm_std.get(age, 0)
        csf_sd = csf_std.get(age, 0)

        dgm_sd_roll = get_roll_dgm(age)
        cgm_sd_roll = get_roll_cgm(age)
        wm_sd_roll = get_roll_wm(age)
        csf_sd_roll = get_roll_csf(age)

        dgm_fit = gam_M_dGM.predict([[age]])[0] if (sex=="M" and dgm_vol is not None) else \
                 gam_F_dGM.predict([[age]])[0] if (sex=="F" and dgm_vol is not None) else None
        cgm_fit = gam_M_cGM.predict([[age]])[0] if  (sex=="M" and cgm_vol is not None) else \
                gam_F_cGM.predict([[age]])[0] if  (sex=="F" and dgm_vol is not None) else None
        wm_fit = gam_M_WM.predict([[age]])[0] if (sex=="M" and wm_vol is not None) else \
                 gam_F_WM.predict([[age]])[0] if (sex=="F" and wm_vol is not None) else None
        csf_fit = gam_M_CSF.predict([[age]])[0] if (sex=="M" and csf_vol is not None) else \
                  gam_F_CSF.predict([[age]])[0] if (sex=="F" and csf_vol is not None) else None

        # Normal z-scores
        dgm_z = (dgm_vol - dgm_fit) / dgm_sd if (dgm_vol is not None and dgm_fit is not None and dgm_sd != 0) else None
        cgm_z = (cgm_vol - cgm_fit) / cgm_sd if (cgm_vol is not None and cgm_fit is not None and cgm_sd != 0) else None
        wm_z = (wm_vol - wm_fit) / wm_sd if (wm_vol is not None and wm_fit is not None and wm_sd != 0) else None
        csf_z = (csf_vol - csf_fit) / csf_sd if (csf_vol is not None and csf_fit is not None and csf_sd != 0) else None

        # Rolling z-scores
        dgm_z_roll = (dgm_vol - dgm_fit) / dgm_sd_roll if (dgm_vol is not None and dgm_fit is not None and dgm_sd_roll != 0) else None
        cgm_z_roll = (cgm_vol - cgm_fit) / cgm_sd_roll if (cgm_vol is not None and cgm_fit is not None and cgm_sd_roll != 0) else None
        wm_z_roll = (wm_vol - wm_fit) / wm_sd_roll if (wm_vol is not None and wm_fit is not None and wm_sd_roll != 0) else None
        csf_z_roll = (csf_vol - csf_fit) / csf_sd_roll if (csf_vol is not None and csf_fit is not None and csf_sd_roll != 0) else None

        # Mean of the three normal z-scores
        normal_vals = [x for x in [dgm_z, cgm_z, wm_z, csf_z] if x is not None]
        mean_z = sum(normal_vals)/len(normal_vals) if normal_vals else None

        # Mean of the three rolling-based z-scores
        roll_vals = [x for x in [dgm_z_roll, cgm_z_roll, wm_z_roll, csf_z_roll] if x is not None]
        mean_z_roll = sum(roll_vals)/len(roll_vals) if roll_vals else None

        row = {
            "subject": subj,
            "age": age,
            "sex": sex,
            "zscore_dgm": dgm_z,
            "zscore_cgm": cgm_z,
            "zscore_wm": wm_z,
            "zscore_csf": csf_z,
            "mean_zscore": mean_z,
            "zscore_dgm_rolling": dgm_z_roll,
            "zscore_cgm_rolling": cgm_z_roll,
            "zscore_wm_rolling": wm_z_roll,
            "zscore_csf_rolling": csf_z_roll,
            "mean_zscore_rolling": mean_z_roll,
            "excluded_seg": ""
        }
        rows_raw.append(row)

    # Adding columns subject_mean_roll_z, difference_z, n_sessions
    rows_by_subject = defaultdict(list)
    for row in rows_raw:
        rows_by_subject[row["subject"]].append(row)

    for subj, subj_rows in rows_by_subject.items():
        n_ses = len(subj_rows)
        valid_overall = [r["mean_zscore_rolling"] for r in subj_rows if r["mean_zscore_rolling"] is not None]
        subj_mean_roll_z = sum(valid_overall)/len(valid_overall) if valid_overall else None

        valid_dgm = [r["zscore_dgm_rolling"] for r in subj_rows if r["zscore_dgm_rolling"] is not None]
        valid_cgm = [r["zscore_cgm_rolling"] for r in subj_rows if r["zscore_cgm_rolling"] is not None]
        valid_wm  = [r["zscore_wm_rolling"]  for r in subj_rows if r["zscore_wm_rolling"] is not None]
        valid_csf = [r["zscore_csf_rolling"] for r in subj_rows if r["zscore_csf_rolling"] is not None]

        subj_mean_roll_z_dgm = sum(valid_dgm)/len(valid_dgm) if valid_dgm else None
        subj_mean_roll_z_cgm = sum(valid_cgm)/len(valid_cgm) if valid_cgm else None
        subj_mean_roll_z_wm  = sum(valid_wm)/len(valid_wm)   if valid_wm  else None
        subj_mean_roll_z_csf = sum(valid_csf)/len(valid_csf) if valid_csf else None

        subj_rows.sort(key=lambda r: r["age"])

        if n_ses == 1:
            diff_val = subj_rows[0]["mean_zscore_rolling"]
            subj_rows[0]["difference_z"] = diff_val
            subj_rows[0]["subject_mean_roll_z"] = subj_mean_roll_z
            subj_rows[0]["subject_mean_roll_z_dgm"] = subj_mean_roll_z_dgm
            subj_rows[0]["subject_mean_roll_z_cgm"] = subj_mean_roll_z_cgm
            subj_rows[0]["subject_mean_roll_z_wm"]  = subj_mean_roll_z_wm
            subj_rows[0]["subject_mean_roll_z_csf"] = subj_mean_roll_z_csf
            subj_rows[0]["n_sessions"] = 1

        elif n_ses == 2:
            z1 = subj_rows[0]["mean_zscore_rolling"]
            z2 = subj_rows[1]["mean_zscore_rolling"]
            if z1 is not None and z2 is not None:
                diff_val = z2 - z1
            else:
                diff_val = None
            for r in subj_rows:
                r["difference_z"] = diff_val
                r["subject_mean_roll_z"] = subj_mean_roll_z
                r["subject_mean_roll_z_dgm"] = subj_mean_roll_z_dgm
                r["subject_mean_roll_z_cgm"] = subj_mean_roll_z_cgm
                r["subject_mean_roll_z_wm"]  = subj_mean_roll_z_wm
                r["subject_mean_roll_z_csf"] = subj_mean_roll_z_csf
                r["n_sessions"] = 2

        else: # n_ses >= 3
            for r in subj_rows:
                if r["mean_zscore_rolling"] is not None and subj_mean_roll_z is not None:
                    diff_val = r["mean_zscore_rolling"] - subj_mean_roll_z
                else:
                    diff_val = None
                r["difference_z"] = diff_val
                r["subject_mean_roll_z"] = subj_mean_roll_z
                r["subject_mean_roll_z_dgm"] = subj_mean_roll_z_dgm
                r["subject_mean_roll_z_cgm"] = subj_mean_roll_z_cgm
                r["subject_mean_roll_z_wm"]  = subj_mean_roll_z_wm
                r["subject_mean_roll_z_csf"] = subj_mean_roll_z_csf
                r["n_sessions"] = n_ses

    updated_rows = []
    for subj, subj_rows in rows_by_subject.items():
        updated_rows.extend(subj_rows)

    # Write out the CSV
    if csv_file:
        output_csv=f'/path/to/save/{version_name}_zscore.csv'
        header = [
            "subject", "age", "sex",
            "zscore_dgm", "zscore_cgm", "zscore_wm", "zscore_csf", "mean_zscore",
            "zscore_dgm_rolling", "zscore_cgm_rolling", "zscore_wm_rolling", "zscore_csf_rolling", "mean_zscore_rolling",
            "subject_mean_roll_z",  "subject_mean_roll_z_dgm", "subject_mean_roll_z_cgm", "subject_mean_roll_z_wm", "subject_mean_roll_z_csf", "difference_z", "n_sessions", "excluded_seg"
        ]

        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in updated_rows:
                writer.writerow([
                    row["subject"],
                    row["age"],
                    row["sex"],
                    row["zscore_dgm"],
                    row["zscore_cgm"],
                    row["zscore_wm"],
                    row["zscore_csf"],
                    row["mean_zscore"],
                    row["zscore_dgm_rolling"],
                    row["zscore_cgm_rolling"],
                    row["zscore_wm_rolling"],
                    row["zscore_csf_rolling"],
                    row["mean_zscore_rolling"],
                    row.get("subject_mean_roll_z"),
                    row.get("subject_mean_roll_z_dgm"),
                    row.get("subject_mean_roll_z_cgm"),
                    row.get("subject_mean_roll_z_wm"),
                    row.get("subject_mean_roll_z_csf"),
                    row.get("difference_z"),
                    row.get("n_sessions"),
                    row["excluded_seg"]
                ])
    
    return updated_rows

if __name__ == "__main__":

    with open('path/to/dict/dict_pred.pkl', 'rb') as f:
        loaded_subject_data = pickle.load(f)
    gam_M_dGM, gam_F_dGM = plot_population(loaded_subject_data, tissue='deep_GM')
    gam_M_cGM, gam_F_cGM = plot_population(loaded_subject_data, tissue='cortical_GM')
    gam_M_WM, gam_F_WM = plot_population(loaded_subject_data, tissue='WM')
    gam_M_CSF, gam_F_CSF = plot_population(loaded_subject_data, tissue='CSF')
        
    compute_zscore(
        loaded_subject_data,
        gam_M_dGM, gam_F_dGM,
        gam_M_cGM, gam_F_cGM,
        gam_M_WM, gam_F_WM,
        gam_M_CSF, gam_F_CSF,
        version_name='pred',
        max_age=None,
        rolling_window=15,
        csv_file=True
    )

