import numpy as np
import nibabel as nib
import scipy.ndimage
import os

def generate_gmm_image(data, means, stds):
    """
    Generate an image by applying voxel intensities from Gaussian distributions to a label map.

    Args:
    - data: 3D numpy array with the segmented labels.
    - means: List or numpy array of mean values corresponding to each label.
    - std_devs: List or numpy array of standard deviations corresponding to each label.

    Returns:
    - new_image: 3D numpy array with the generated values from the GMM.
    """

    labels_to_category = {
    label: category
    for category, labels in {
        0: [0],                          # Background
        1: [24],                         # CSF
        2: [16],                         # Brainstem
        3: [4, 5, 14, 15, 43, 44],       # Ventricles
        4: [2, 41],                      # WM
        5: [12, 51],                     # Putamen
        6: [3, 42],                      # Cerebral Cortex
        7: [7, 46],                      # Cerebellum WM
        8: [8, 47],                      # Cerebellum Cortex
        9: [10, 49],                     # Thalamus
        10: [11, 50],                    # Caudate
        11: [13, 52],                    # Pallidum
        12: [17, 53],                    # Hippocampus
        13: [18, 54],                    # Amygdala
        14: [26, 58],                    # Accumbens Area
        15: [28, 60]                     # Ventral DC
    }.items()
    for label in labels
    }

    new_image = np.zeros_like(data, dtype=np.float32)

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                label = data[x, y, z]

                if label not in labels_to_category:
                    raise ValueError(f"Label {label} not found in 'labels_to_category'.")

                cat_idx = labels_to_category[label]

                new_image[x, y, z] = np.random.normal(
                    loc=means[cat_idx],
                    scale=stds[cat_idx]
                )
    return new_image

def downsample(image, resolution, current_resolution):

    downsample_factor = (current_resolution[0] / resolution[0],
                         current_resolution[1] / resolution[1],
                         current_resolution[2] / resolution[2])

    downsampled_image = scipy.ndimage.zoom(image, zoom=downsample_factor, order=1)
    upsampled_image = scipy.ndimage.zoom(downsampled_image, zoom=(1/downsample_factor[0], 1/downsample_factor[1], 1/downsample_factor[2]), order=1)
    return upsampled_image

def blurring_image(image, sigma):

    blur_image = scipy.ndimage.gaussian_filter(image, sigma=sigma)
    return blur_image

def add_noise(image, noise_std):

    noise = np.random.normal(0, noise_std, image.shape)
    noisy_image = image + noise
    return noisy_image


def create_synthetic_image (segmentation, means, stds, output_dir, current_resolution, sigma=1.15, resolution=(2.0, 2.0, 2.0), output_name='_synthetic.nii.gz'):
    """"
    Creates the ultra-low-field synthetic image by applying degradation to the Gaussian image.

    Args:
    - segmentation: Path to a Nifti file of a brain segmentation.
    - means: List or numpy array of mean values corresponding to each label.
    - stds: List or numpy array of standard deviations corresponding to each label.
    - output_file: Path to the directory where the synthetic image will be saved.
    - current_resolution: Tuple (x, y, z) with the current resolution of the image.
    - sigma: Standard deviation of the Gaussian filter.
    - resolution: Tuple (x, y, z) with the target resolution.

    Returns:
    - new_image: 3D numpy array with the synthetic image
    """

    segmentation_img = nib.load(segmentation)
    data = segmentation_img.get_fdata()
    data = np.round(data).astype(np.int32)

    gmm_image = generate_gmm_image(data, means, stds)
    downsample_im = downsample(gmm_image, resolution=resolution, current_resolution=current_resolution)
    blur_im = blurring_image(downsample_im, sigma=sigma)
    final_im = add_noise(blur_im, noise_std=(36.8/1000))
    
    new_nifti_image = nib.Nifti1Image(final_im, segmentation_img.affine)
    output_file = os.path.join(output_dir, output_name)
    nib.save(new_nifti_image, output_file)

    return final_im

