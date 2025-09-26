import numpy as np
import pickle
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from plot_T1values import rolling_mean, compute_mean_value_arrays
import matplotlib.pyplot as plt
import pandas as pd

def create_region_arrays(dict_path):
    """
    Creates the arrays of priors for each month and saves them as numpy arrays.
    Each array holds the values for the 16 regions.

    """

    with open(dict_path, 'rb') as handle:
        data = pickle.load(handle)

    x_values = np.arange(2, 82)
    region_arrays = [np.zeros(16) for _ in range(len(x_values))]

    regions = {
        "brainstem":        {"key": "posterior_pons", "idx": 2,  "offset": 0.0},
        "thalamus":         {"key": "thalamus",       "idx": 9,  "offset": 0.0},
        "putamen":          {"key": "putamen",        "idx": 5,  "offset": 0.0},
        "white_matter":     {"key": "frontalWM",      "idx": 4,  "offset": 0.0},
        "caudate_nucleus":  {"key": "caudnl",         "idx": 10, "offset": 0.0},
        "cerebral_cortex":  {"key": None,             "idx": 6,  "offset": 0.0},
        "cerebellum_wm":    {"key": "frontalWM",      "idx": 7,  "offset": 0.0},
        "cerebellum_cortex":{"key": None,             "idx": 8,  "offset": 0.0},
        "pallidum":         {"key": "caudnl",         "idx": 11, "offset": 0.0},
        "hippocampus":      {"key": "caudnl",         "idx": 12, "offset": 0.0},
        "amygdala":         {"key": "caudnl",         "idx": 13, "offset": 0.0},
        "accumbens_area":   {"key": "caudnl",         "idx": 14, "offset": 0.0},
        "ventral_dc":       {"key": "caudnl",         "idx": 15, "offset": 0.0},
        "csf":              {"key": None,             "idx": 1,  "offset": 0.0},
        "ventricles":       {"key": None,             "idx": 3,  "offset": 0.0},
        "background":       {"key": None,             "idx": 0,  "offset": 0.0},
    }

    if "caudnl" in data:
            full_x = np.arange(2, 82)
            y_caudnl = np.array(data["caudnl"], dtype=float)
            Y_rolling_caudnl = rolling_mean(full_x, y_caudnl, window_size=25)
            X_unique, uniq_idx = np.unique(full_x, return_index=True)
            Y_caudnl_clean = Y_rolling_caudnl[uniq_idx]

            caudnl_spline = UnivariateSpline(X_unique, Y_caudnl_clean, s=2.0)

            # Evaluate the mean of the caudnl spline from 40-62mo to compute the offsets
            x_eval = np.arange(40, 63)
            y_eval = caudnl_spline(x_eval)
            mean_caudnl_40_62 = np.mean(y_eval)
            print("Mean caudnl 40-62:", mean_caudnl_40_62)

    if "frontalWM" in data:
        full_x = np.arange(2, 82)
        y_frontalWM = np.array(data["frontalWM"], dtype=float)
        Y_rolling_frontalWM = rolling_mean(full_x, y_frontalWM, window_size=25)
        X_unique, uniq_idx = np.unique(full_x, return_index=True)
        Y_frontalWM_clean = Y_rolling_frontalWM[uniq_idx]

        frontalWM_spline = UnivariateSpline(X_unique, Y_frontalWM_clean, s=2.0)

        # Evaluate the mean of the frontalWM spline from 40-62mo to compute the offsets
        x_eval = np.arange(40, 63)
        y_eval = frontalWM_spline(x_eval)
        mean_frontalWM_40_62 = np.mean(y_eval)
        print("Mean frontalWM 40-62:", mean_frontalWM_40_62)


    # Regions that need an added offset
    caudnl_regions = ["pallidum","hippocampus","amygdala","accumbens_area","ventral_dc"]
    frontalWM_regions = ["cerebellum_wm"]
    cortex_regions = ["cerebral_cortex", "cerebellum_cortex"]

    for region_name, info in regions.items():
        region_idx = info["idx"]
        data_key = info["key"]
        base_offset = info["offset"]

        # Add the offsets to the dictionary
        if region_name in caudnl_regions:
            offset = (compute_mean_value_arrays(region_idx)/1000) - mean_caudnl_40_62
            print(f"Offset caudnl for {region_name}:", offset)
        
        elif region_name in frontalWM_regions:
            offset = (compute_mean_value_arrays(region_idx)/1000) - mean_frontalWM_40_62
            print(f"Offset frontalWM for {region_name}:", offset)

        elif region_name in cortex_regions:
            # Compute the offset with the spline created with the cortical values 
            # from "Cortical maturation and myelination in healthy toddlers and young children (Deoni, 2015)"
            age = np.array([300, 800, 1300, 1800, 2300]) / 30.4375 # Converting days to months
            t1 = np.array([2100, 2000, 1950, 1900, 1850]) / 1000.0 # Converting ms to seconds
            cortex_spline = UnivariateSpline(age, t1, s=2.0)
            
            x_eval = np.arange(40, 63)
            y_eval = cortex_spline(x_eval)
            mean_cortex_40_62 = np.mean(y_eval)

            offset = (compute_mean_value_arrays(region_idx)/1000) - mean_cortex_40_62 - 0.25
            print(f"Offset cortex for {region_name}:", offset)
        else:
            offset = base_offset

        # Assign the values to the regions that don't follow a spline
        if data_key is None:
            if region_name in ["csf", "ventricles"]:
                val = compute_mean_value_arrays(region_idx) / 1000
                print("Mean value for", region_name, ":", val)
                for i in range(len(x_values)):
                    region_arrays[i][region_idx] = val
                    
            elif region_name == "background":
                for i in range(len(x_values)):
                    region_arrays[i][region_idx] = 0

            elif region_name in ["cerebral_cortex", "cerebellum_cortex"]:
                age = np.array([300, 800, 1300, 1800, 2300]) / 30.4375
                t1_inf_temp = np.array([2100, 2000, 1950, 1900, 1850]) / 1000
                t1_precent = np.array([2050, 1950, 1900, 1850, 1800]) / 1000 
                t1_inf_front = np.array([2300, 2200, 2100, 2050, 2000]) / 1000 
                t1 = (t1_inf_temp + t1_precent + t1_inf_front) / 3.0

                cortex_spline = UnivariateSpline(age, t1, s=2.0)
                spline_values = cortex_spline(x_values) + offset

                for i in range(len(x_values)):
                    region_arrays[i][region_idx] = spline_values[i]

        else:
            y = np.array(data[data_key], dtype=float)
            Y_rolling = rolling_mean(x_values, y, window_size=25)
            X_unique, uniq_idx = np.unique(x_values, return_index=True)
            Y_clean = Y_rolling[uniq_idx]

            rolling_spline = UnivariateSpline(X_unique, Y_clean, s=2.0)

            spline_values = rolling_spline(x_values) + offset

            for i, x_val in enumerate(x_values):
                region_arrays[i][region_idx] = spline_values[i]
    
    for i, arr in enumerate(region_arrays):
       np.save(f"path/to/arrays/region_array_{i+2}.npy", arr) # Save the arrays with the age in months, starting at 2 months

    return region_arrays
