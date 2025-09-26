import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import nibabel as nib
import numpy as np
import pandas as pd
import glob

def load_test_data(path_test, csv_file=True):
    """
    Loads test data either from a folder of NIfTI files or from 
    a CSV file with columns [subject, session, path_3T, path_LF]
    """
    if csv_file:
        df = pd.read_csv(path_test)
        df_filtered = df[df["path_LF"].str.endswith(".nii.gz")]
        df_filtered = df_filtered.reset_index(drop=True)
        nb_images_test = len(df_filtered)

        test_data = np.zeros((nb_images_test, 192, 192, 160, 1), dtype=np.float32)
        affines = []
        file_names = []

        for i, row in df_filtered.iterrows():
            lf_path = row["path_LF"]
            img = nib.load(lf_path)
            img_data = img.get_fdata() 

            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
            img_data = img_data[..., np.newaxis]
            test_data[i] = img_data
            affines.append(img.affine)
            file_names.append(os.path.basename(lf_path))

        affines = np.array(affines)
        return test_data, affines, file_names
    
    else:
        nii_files_x_test = [f for f in os.listdir(path_test) if f.endswith('.nii.gz')]
        nb_images_test = len(nii_files_x_test)

        test_data = np.zeros((nb_images_test, 192, 192, 160, 1), dtype=np.float32)
        affines = np.zeros((nb_images_test, 4, 4))
        file_names = []

        for i, file_name in enumerate(sorted(nii_files_x_test)):
            img_path = os.path.join(path_test, file_name)
            img = nib.load(img_path)
            img_data = img.get_fdata()

            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
            img_data = img_data[..., np.newaxis]
            test_data[i] = img_data
            affines[i] = img.affine
            file_names.append(file_name)

        return test_data, affines, file_names

def predict_images(path_test_x, model_path, out_pred_dir, csv_file=False):

    test_data, affines, file_names = load_test_data(path_test_x, csv_file=csv_file)
    model = load_model(model_path)
    if not os.path.exists(out_pred_dir):
        os.makedirs(out_pred_dir)

    for i in range(len(test_data)):
        input_data = np.expand_dims(test_data[i], axis=0)
        image = model.predict(input_data, verbose=1)
        image_3d = image.squeeze(axis=0)
        image_3d = image_3d.squeeze(axis=-1)

        image_3d = np.nan_to_num(image_3d, nan=0, posinf=0, neginf=0)
        nifti_image = nib.Nifti1Image(image_3d, affine=affines[i])
        out_file_name = f"pred_{file_names[i]}"
        out_file_path = os.path.join(out_pred_dir, out_file_name)
        nib.save(nifti_image, out_file_path)

        print(f"Saved prediction for {file_names[i]} to {out_file_path}")

