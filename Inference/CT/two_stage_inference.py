import os
import time

import SimpleITK as sitk
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from utils.utils import split_connected_components, remove_small_connected_components, relabel_connected_components, \
    process_final_label, refine_labels


def create_and_apply_mask(volume, prediction, label, padding=10, use_mask: bool = True):
    """
    Create a mask for a specific label and apply it to the volume, keeping only the regions with the specified label.
    Also crop the volume to the bounding box of the mask.
    """
    mask = (prediction == label).astype(np.float32)
    if use_mask:
        volume = volume * mask

    # Find the bounding box of the mask
    coords = np.array(np.nonzero(mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1) + 1

    # Apply padding to the bounding box
    top_left = np.maximum(top_left - padding, 0)
    bottom_right = np.minimum(bottom_right + padding, np.array(volume.shape))

    # Crop the volume to the bounding box of the mask
    cropped_volume = volume[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1],
                     top_left[2]:bottom_right[2]]
    return cropped_volume, top_left, bottom_right


def resample_image(image, new_spacing, interpolator=sitk.sitkLinear):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetInterpolator(interpolator)

    resampled_image = resample.Execute(image)
    return resampled_image


def adjust_shape(a, b):
    """
    Adjust the shape of 3D matrix a to match the shape of 3D matrix b by padding or cropping.

    Parameters:
    a (np.ndarray): The input 3D matrix to be adjusted.
    b (np.ndarray): The reference 3D matrix whose shape we want to match.

    Returns:
    np.ndarray: The adjusted 3D matrix a with the same shape as b.
    """
    # Get the shapes of a and b
    a_shape = a.shape
    b_shape = b.shape

    # Initialize the adjusted array
    adjusted_a = a.copy()

    # Pad or crop each dimension to match the shape of b
    for dim in range(3):
        if a_shape[dim] < b_shape[dim]:
            # Padding
            pad_width = b_shape[dim] - a_shape[dim]
            pad_before = pad_width // 2
            pad_after = pad_width - pad_before
            pad_tuple = [(0, 0), (0, 0), (0, 0)]
            pad_tuple[dim] = (pad_before, pad_after)
            adjusted_a = np.pad(adjusted_a, pad_tuple, mode='constant', constant_values=0)
        elif a_shape[dim] > b_shape[dim]:
            # Cropping
            crop_before = (a_shape[dim] - b_shape[dim]) // 2
            crop_after = crop_before + b_shape[dim]
            if dim == 0:
                adjusted_a = adjusted_a[crop_before:crop_after, :, :]
            elif dim == 1:
                adjusted_a = adjusted_a[:, crop_before:crop_after, :]
            else:
                adjusted_a = adjusted_a[:, :, crop_before:crop_after]

    return adjusted_a


def restore_cropped_volume(cropped_volume, original_shape, top_left, bottom_right):
    """
    Restore the cropped volume to its original shape using the recorded bounding box.
    """
    restored_volume = np.zeros(original_shape, dtype=cropped_volume.dtype)
    restored_volume[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1],
    top_left[2]:bottom_right[2]] = cropped_volume
    return restored_volume


def save_nifti(data, spacing, direction, origin, output_path):
    """
    Save the numpy array as a NIfTI file.
    """
    img = sitk.GetImageFromArray(data.astype(np.int8))
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    img.SetOrigin(origin)
    sitk.WriteImage(img, output_path, useCompression=True)


def inference_one_image(input_dir, output_dir):
    start_time = time.time()
    model_stage0 = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                   perform_everything_on_device=True, device=torch.device('cuda'), verbose=False,
                                   verbose_preprocessing=False, allow_tqdm=False
                                   )
    model_stage0.initialize_from_trained_model_folder(
        'models/Dataset198_PelvicLowres/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres',
        use_folds=(0,), checkpoint_name='checkpoint_best.pth',
    )
    volume = sitk.ReadImage(input_dir)
    # Record raw image information
    old_spacing = volume.GetSpacing()
    old_direction = volume.GetDirection()
    old_origin = volume.GetOrigin()
    old_props = {
        "sitk_stuff":
            {
                'spacing': old_spacing,
                'origin': old_origin,
                'direction': old_direction
            },
        'spacing': [old_spacing[2], old_spacing[1], old_spacing[0]]
    }
    # Resampling to spacing used in training stage0 model
    lowres_volume = resample_image(volume, [2.5, 2.5, 2.5], sitk.sitkLinear)
    # Record the image information after resampling
    new_spacing = lowres_volume.GetSpacing()
    new_direction = lowres_volume.GetDirection()
    new_origin = lowres_volume.GetOrigin()
    new_props = {
        "sitk_stuff":
            {
                'spacing': new_spacing,
                'origin': new_origin,
                'direction': new_direction
            },
        'spacing': [new_spacing[2], new_spacing[1], new_spacing[0]]
    }
    lowres_volume = sitk.GetArrayFromImage(lowres_volume)

    # start stage 1 inference
    stage0_start_time = time.time()

    prediction_stage1 = model_stage0.predict_single_npy_array(np.expand_dims(lowres_volume, axis=0), new_props,
                                                              None, None, False)
    prediction_stage1 = relabel_connected_components(prediction_stage1)
    print(f"Stage 0 inference completed in {time.time() - stage0_start_time:.2f} seconds.")

    # Sample the label back to its original size
    prediction_stage1 = sitk.GetImageFromArray(prediction_stage1)
    prediction_stage1.SetOrigin(new_origin)
    prediction_stage1.SetDirection(new_direction)
    prediction_stage1.SetSpacing(new_spacing)
    prediction_stage1 = resample_image(prediction_stage1, old_spacing, sitk.sitkNearestNeighbor)
    prediction_stage1 = sitk.GetArrayFromImage(prediction_stage1)

    # Remove small segmentations
    prediction_stage1 = remove_small_connected_components(prediction_stage1, [400, 200, 200], [1, 2, 3])

    model_stage1 = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                   perform_everything_on_device=True, device=torch.device('cuda'), verbose=False,
                                   verbose_preprocessing=False, allow_tqdm=False
                                   )
    model_stage1.initialize_from_trained_model_folder(
        'models/Dataset199_Pelvic/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres',
        use_folds=(0,), checkpoint_name='checkpoint_best.pth',
    )


    volume = sitk.GetArrayFromImage(volume)
    # Adjust the interpolated prediction so that its shape is consistent with the original image
    prediction_stage1 = adjust_shape(prediction_stage1, volume)

    # create mask for stage 2 inference
    sacrum_input, sacrum_top_left, sacrum_bottom_right = create_and_apply_mask(volume, prediction_stage1, 1, padding=5, use_mask=False)
    pred_sacrum_stage1 = model_stage1.predict_single_npy_array(np.expand_dims(sacrum_input, axis=0), old_props, None, None, False)
    pred_sacrum_stage1 = np.where(pred_sacrum_stage1 == 1, 1, 0)


    left_input, left_top_left, left_bottom_right = create_and_apply_mask(volume, prediction_stage1, 2, padding=5, use_mask=False)
    pred_left_stage1 = model_stage1.predict_single_npy_array(np.expand_dims(left_input, axis=0), old_props, None, None, False)
    pred_left_stage1 = np.where(pred_left_stage1 == 2, 2, 0)

    right_input, right_top_left, right_bottom_right = create_and_apply_mask(volume, prediction_stage1, 3, padding=5, use_mask=False)
    pred_right_stage1 = model_stage1.predict_single_npy_array(np.expand_dims(right_input, axis=0), old_props, None, None, False)
    pred_right_stage1 = np.where(pred_right_stage1 == 3, 3, 0)

    # start stage 2 inference
    step_size = 0.5
    gaussian_flag = True
    mirror_flag = True
    print("Starting stage2 inference...")
    print(f"step size: {step_size}, gaussian flag: {gaussian_flag}, mirror flag: {mirror_flag}")
    model_sacrum = nnUNetPredictor(tile_step_size=step_size, use_gaussian=gaussian_flag, use_mirroring=mirror_flag,
                                   perform_everything_on_device=True, device=torch.device('cuda'), verbose=False,
                                   verbose_preprocessing=False, allow_tqdm=False
                                   )
    model_sacrum.initialize_from_trained_model_folder(
        'models/Dataset202_newSacrum_old/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres',
        use_folds=(0,), checkpoint_name='checkpoint_best.pth',
    )

    model_hips = nnUNetPredictor(tile_step_size=step_size, use_gaussian=gaussian_flag, use_mirroring=mirror_flag,
                                 perform_everything_on_device=True, device=torch.device('cuda'), verbose=False,
                                 verbose_preprocessing=False, allow_tqdm=False
                                 )
    model_hips.initialize_from_trained_model_folder(
        'models/Dataset203_newHips/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres',
        use_folds=(0,), checkpoint_name='checkpoint_final.pth',
    )

    sacrum_start_time = time.time()
    pred_s_1, prob_s_1 = model_sacrum.predict_single_npy_array(np.expand_dims(sacrum_input, axis=0), old_props, None, None, True)
    pred_s_1 = split_connected_components(pred_s_1, 2, 2, min_volume=400)
    prediction_sacrum = refine_labels(pred_sacrum_stage1, pred_s_1)
    prediction_sacrum = restore_cropped_volume(prediction_sacrum, volume.shape, sacrum_top_left, sacrum_bottom_right)
    print(f"Sacrum inference completed in {time.time() - sacrum_start_time:.2f} seconds.")

    left_start_time = time.time()
    pred_l_1, prob_l_1 = model_hips.predict_single_npy_array(np.expand_dims(left_input, axis=0), old_props, None, None, True)
    pred_l_1 = split_connected_components(pred_l_1, 2, 12)
    pred_l_1 = np.where(pred_l_1 == 2, 0, pred_l_1)
    pred_l_1 = np.where(pred_l_1 == 1, 11, pred_l_1)
    prediction_left = refine_labels(pred_left_stage1, pred_l_1)
    prediction_left = restore_cropped_volume(prediction_left, volume.shape, left_top_left, left_bottom_right)
    print(f"Left hip inference completed in {time.time() - left_start_time:.2f} seconds.")

    right_start_time = time.time()
    pred_r_1, prob_r_1 = model_hips.predict_single_npy_array(np.expand_dims(right_input, axis=0), old_props, None, None, True)
    pred_r_1 = split_connected_components(pred_r_1, 2, 22)
    pred_r_1 = np.where(pred_r_1 == 2, 0, pred_r_1)
    pred_r_1 = np.where(pred_r_1 == 1, 21, pred_r_1)
    prediction_right = refine_labels(pred_right_stage1, pred_r_1)
    prediction_right = restore_cropped_volume(prediction_right, volume.shape, right_top_left, right_bottom_right)
    print(f"Right hip inference completed in {time.time() - right_start_time:.2f} seconds.")

    # merge and save labels
    combined_prediction = np.maximum.reduce([prediction_sacrum, prediction_left, prediction_right])
    combined_prediction = process_final_label(combined_prediction)
    combined_prediction = remove_small_connected_components(combined_prediction, [1000, 400, 400], [1, 11, 21])
    assert {1, 11, 21}.issubset(np.unique(combined_prediction)), "something wrong in label processing!"
    input_filename = os.path.basename(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, input_filename)
    save_nifti(combined_prediction, old_spacing, old_direction, old_origin, output_path)
    total_time = time.time() - start_time
    print(f"Total inference time: {total_time:.2f} seconds.")


if __name__ == "__main__":
    input_dir = r"/data/ypy/dataset/miccai_challenge_2024/miccai_challenge_2024_nii/3cases_Preliminary_phase"
    output_dir = r"/data/ypy/dataset/miccai_challenge_2024/miccai_challenge_2024_nii/3cases_Preliminary_phase/mask_0818_refined"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".nii.gz") or filename.endswith(".mha"):
            print("*********************************Processing {}**********************************".format(filename))
            input_path = os.path.join(input_dir, filename)
            # output_path = os.path.join(output_dir, filename)
            inference_one_image(input_path, output_dir)
