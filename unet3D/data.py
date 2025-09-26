import nibabel as nib
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, nii_files_x, nii_files_y, batch_size=2, dim=(192, 192, 160), n_channels=1, shuffle=True):
        self.nii_files_x = nii_files_x
        self.nii_files_y = nii_files_y
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.nii_files_x))
        self.idx_y = np.arange(len(self.nii_files_y))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.nii_files_x) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        return self.__data_generation(idxs)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idxs):
        x_batch = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y_batch = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)

        for i, idx in enumerate(idxs):
            nib_x = nib.load(self.nii_files_x[idx])
            nib_y = nib.load(self.nii_files_y[idx])

            data_x = nib_x.get_fdata(dtype=np.float32)
            data_y = nib_y.get_fdata(dtype=np.float32)

            # Normalize
            data_x = (data_x - np.min(data_x)) / (np.max(data_x) - np.min(data_x))
            data_y = (data_y - np.min(data_y)) / (np.max(data_y) - np.min(data_y))

            x_batch[i, ..., 0] = data_x[: self.dim[0], : self.dim[1], : self.dim[2]]
            y_batch[i, ..., 0] = data_y[: self.dim[0], : self.dim[1], : self.dim[2]]

        return x_batch, y_batch

def load_data_from_csv(fold_index, n_folds=5):
    """
    Loads training and validation data from 5 CSVs.
    The CSV format is:
       1) subject
       2) session
       3) path_y  (3T)
       4) path_x  (LF)
    For the given fold_index (0 to 4):
      - Uses fold_{fold_index+1}.csv as validation data.
      - Uses the other 4 CSVs as training data.
    """
    train_frames = []
    for i in range(n_folds):
        csv_path = f"fold_{i+1}.csv"
        if i != fold_index: 
            train_frames.append(pd.read_csv(csv_path))
        else: 
            val_df = pd.read_csv(csv_path)

    train_df = pd.concat(train_frames, ignore_index=True)

    # Convert train/val dataframes into lists of paths
    train_paths_x = train_df.iloc[:, 3].tolist()
    train_paths_y = train_df.iloc[:, 2].tolist()

    val_paths_x = val_df.iloc[:, 3].tolist()
    val_paths_y = val_df.iloc[:, 2].tolist()

    return train_paths_x, train_paths_y, val_paths_x, val_paths_y

def training_data(path_train_x, path_train_y, path_val_x, path_val_y, batch_size=2, dim=(192, 192, 160)):

    train_generator = DataGenerator(path_train_x, path_train_y, batch_size=batch_size, dim=dim)
    val_generator = DataGenerator(path_val_x, path_val_y, batch_size=batch_size, dim=dim)

    sample_x, sample_y = next(iter(train_generator))
    return sample_x, sample_y, train_generator, val_generator

