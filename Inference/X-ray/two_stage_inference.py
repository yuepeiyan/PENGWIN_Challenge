import os
import time

import numpy as np
import torch
import tifffile as tiff
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from utils.utils import split_connected_components, remove_small_connected_components, refine_labels
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def restore_cropped_volume(cropped_volume, original_shape, top_left, bottom_right):
    """
    Restore the cropped volume to its original shape using the recorded bounding box.
    """
    restored_volume = np.zeros(original_shape, dtype=cropped_volume.dtype)
    restored_volume[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1],
    top_left[2]:bottom_right[2]] = cropped_volume
    return restored_volume


def merge_labels(label_list):
    """
    Merge multiple 2D segmentation labels into a single uint32 label, where each pixel can belong to multiple categories.

    Parameters:
    label_list (list of np.ndarray): A list of labels, each element is a 2D numpy array with values between 0 and 30.

    Returns:
    np.ndarray: The merged label array, with data type uint32.
    """
    # Assume all labels have the same shape, get the shape of the arrays
    shape = label_list[0].shape

    # Create an array of the same size with data type uint32
    combined_label = np.zeros(shape, dtype=np.uint32)

    # Iterate over each label
    for label in label_list:
        # Iterate over each pixel
        for i in range(shape[0]):
            for j in range(shape[1]):
                value = label[i, j]
                if value > 0:  # If the label value is greater than 0, update the corresponding bit
                    combined_label[i, j] |= (1 << value)

    return combined_label



def inference_one_image(input_dir, output_dir):
    start_time = time.time()
    # Load the image to be predicted
    image = tiff.imread(input_dir)
    # Expand the image by one dimension to fit the input of NNUNET
    image = np.expand_dims(image, axis=0)
    props = {
        'spacing': [999.0, 1.0, 1.0]
    }
    # Load three models in stage1
    model_stage1_s = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                     perform_everything_on_device=True, device=torch.device('cuda'), verbose=False,
                                     verbose_preprocessing=False, allow_tqdm=False
                                     )
    model_stage1_s.initialize_from_trained_model_folder(
        'models/Dataset197_xray_sacrum/nnUNetTrainer__nnUNetResEncUNetMPlans__2d',
        use_folds=(0,), checkpoint_name='checkpoint_final.pth',
    )
    model_stage1_l = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                     perform_everything_on_device=True, device=torch.device('cuda'), verbose=False,
                                     verbose_preprocessing=False, allow_tqdm=False
                                     )
    model_stage1_l.initialize_from_trained_model_folder(
        'models/Dataset195_xray_left/nnUNetTrainerPelvic__nnUNetResEncUNetMPlans__2d',
        use_folds=(0,), checkpoint_name='checkpoint_best.pth',
    )
    model_stage1_r = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                     perform_everything_on_device=True, device=torch.device('cuda'), verbose=False,
                                     verbose_preprocessing=False, allow_tqdm=False
                                     )
    model_stage1_r.initialize_from_trained_model_folder(
        'models/Dataset196_xray_right/nnUNetTrainerPelvic__nnUNetResEncUNetMPlans__2d',
        use_folds=(0,), checkpoint_name='checkpoint_best.pth',
    )

    # # start stage 1 inference
    stage1_start_time = time.time()
    pred_stage1_s = model_stage1_s.predict_single_npy_array(np.expand_dims(image, axis=0), props, None, None, False)
    pred_stage1_l = model_stage1_l.predict_single_npy_array(np.expand_dims(image, axis=0), props, None, None, False)
    pred_stage1_r = model_stage1_r.predict_single_npy_array(np.expand_dims(image, axis=0), props, None, None, False)

    print(f"Stage 1 inference completed in {time.time() - stage1_start_time:.2f} seconds.")

    # start stage 2 inference
    step_size = 0.5
    gaussian_flag = True
    mirror_flag = True
    # load stage2 models
    model_stage2_s_1 = nnUNetPredictor(tile_step_size=step_size, use_gaussian=gaussian_flag,
                                       use_mirroring=mirror_flag,
                                       perform_everything_on_device=True, device=torch.device('cuda'),
                                       verbose=False,
                                       verbose_preprocessing=False, allow_tqdm=False
                                       )
    model_stage2_s_1.initialize_from_trained_model_folder(
        'models/Dataset193_xray_sacrum1/nnUNetTrainer__nnUNetResEncUNetMPlans__2d',
        use_folds=(0,), checkpoint_name='checkpoint_final.pth',
    )
    model_stage2_s_2 = nnUNetPredictor(tile_step_size=step_size, use_gaussian=gaussian_flag,
                                       use_mirroring=mirror_flag,
                                       perform_everything_on_device=True, device=torch.device('cuda'),
                                       verbose=False,
                                       verbose_preprocessing=False, allow_tqdm=False
                                       )
    model_stage2_s_2.initialize_from_trained_model_folder(
        'models/Dataset194_xray_sacrum2/nnUNetTrainer__nnUNetResEncUNetMPlans__2d',
        use_folds=(0,), checkpoint_name='checkpoint_final.pth',
    )

    model_stage2_hip_1 = nnUNetPredictor(tile_step_size=step_size, use_gaussian=gaussian_flag,
                                       use_mirroring=mirror_flag,
                                       perform_everything_on_device=True, device=torch.device('cuda'),
                                       verbose=False,
                                       verbose_preprocessing=False, allow_tqdm=False
                                       )
    model_stage2_hip_1.initialize_from_trained_model_folder(
        'models/Dataset187_xray_hips1/nnUNetTrainer__nnUNetResEncUNetMPlans__2d',
        use_folds=(0,), checkpoint_name='checkpoint_final.pth',
    )

    model_stage2_hip_2 = nnUNetPredictor(tile_step_size=step_size, use_gaussian=gaussian_flag,
                                       use_mirroring=mirror_flag,
                                       perform_everything_on_device=True, device=torch.device('cuda'),
                                       verbose=False,
                                       verbose_preprocessing=False, allow_tqdm=False
                                       )
    model_stage2_hip_2.initialize_from_trained_model_folder(
        'models/Dataset188_xray_hips2/nnUNetTrainer__nnUNetResEncUNetMPlans__2d',
        use_folds=(0,), checkpoint_name='checkpoint_best.pth',
    )

    label_list = []
    if pred_stage1_s.any():
        sacrum_start_time = time.time()
        sacrum_input, sacrum_top_left, sacrum_bottom_right = create_and_apply_mask(image, pred_stage1_s, 1, padding=5, use_mask=False)

        pred_s_1, prob_s_1 = model_stage2_s_1.predict_single_npy_array(np.expand_dims(sacrum_input, axis=0), props, None, None, True)
        pred_s_1 = remove_small_connected_components(pred_s_1, [10], [1])
        pred_s_1 = restore_cropped_volume(pred_s_1, image.shape, sacrum_top_left, sacrum_bottom_right)
        pred_s_1 = refine_labels(pred_stage1_s, pred_s_1)
        label_list.append(pred_s_1)

        pred_s_2, prob_s_2 = model_stage2_s_2.predict_single_npy_array(np.expand_dims(sacrum_input, axis=0), props, None, None, True)
        pred_s_2 = restore_cropped_volume(pred_s_2, image.shape, sacrum_top_left, sacrum_bottom_right)
        pred_s_2 = refine_labels(pred_stage1_s, pred_s_2)
        pred_s_2 = remove_small_connected_components(pred_s_2, [10], [1])
        pred_s_2 = split_connected_components(pred_s_2, 1, 2)
        label_list.append(pred_s_2)
        print(f"Sacrum inference completed in {time.time() - sacrum_start_time:.2f} seconds.")

    if pred_stage1_l.any():
        left_start_time = time.time()
        left_input, left_top_left, left_bottom_right = create_and_apply_mask(image, pred_stage1_l, 1, padding=5, use_mask=False)

        pred_l_1, prob_l_1 = model_stage2_hip_1.predict_single_npy_array(np.expand_dims(left_input, axis=0), props, None, None, True)
        pred_l_1 = remove_small_connected_components(pred_l_1, [10], [1])
        pred_l_1 = np.where(pred_l_1 == 1, 11, pred_l_1)
        pred_l_1 = restore_cropped_volume(pred_l_1, image.shape, left_top_left, left_bottom_right)
        label_list.append(pred_l_1)

        pred_l_2, prob_l_2 = model_stage2_hip_2.predict_single_npy_array(np.expand_dims(left_input, axis=0), props, None, None, True)
        pred_l_2 = remove_small_connected_components(pred_l_2, [10], [1])
        pred_l_2 = split_connected_components(pred_l_2, 1, 12)
        pred_l_2 = restore_cropped_volume(pred_l_2, image.shape, left_top_left, left_bottom_right)
        label_list.append(pred_l_2)
        print(f"Left hip inference completed in {time.time() - left_start_time:.2f} seconds.")

    if pred_stage1_r.any():
        right_start_time = time.time()
        right_input, right_top_left, right_bottom_right = create_and_apply_mask(image, pred_stage1_r, 1, padding=5, use_mask=False)

        pred_r_1, prob_r_1 = model_stage2_hip_1.predict_single_npy_array(np.expand_dims(right_input, axis=0), props, None, None, True)
        pred_r_1 = remove_small_connected_components(pred_r_1, [10], [1])
        pred_r_1 = np.where(pred_r_1 == 1, 21, pred_r_1)
        pred_r_1 = restore_cropped_volume(pred_r_1, image.shape, right_top_left, right_bottom_right)
        label_list.append(pred_r_1)

        pred_r_2, prob_r_2 = model_stage2_hip_2.predict_single_npy_array(np.expand_dims(right_input, axis=0), props, None, None, True)
        pred_r_2 = remove_small_connected_components(pred_r_2, [10], [1])
        pred_r_2 = split_connected_components(pred_r_2, 1, 22)
        pred_r_2 = restore_cropped_volume(pred_r_2, image.shape, right_top_left, right_bottom_right)
        label_list.append(pred_r_2)
        print(f"Right hip inference completed in {time.time() - right_start_time:.2f} seconds.")

    label_list = [np.squeeze(label, axis=0) for label in label_list]
    combined_prediction = merge_labels(label_list)
    input_filename = os.path.basename(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, input_filename)
    with tiff.TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(combined_prediction, photometric='minisblack', metadata={'spacing': 1, 'unit': 'um'}, resolution=(1, 1, 'CENTIMETER'))
    total_time = time.time() - start_time
    print(f"Total inference time: {total_time:.2f} seconds.")


if __name__ == "__main__":
    input_dir = r"/data/ypy/dataset/miccai_challenge_2024/miccai_challenge_2024_nii/xray_10case"
    output_dir = r"/data/ypy/dataset/miccai_challenge_2024/miccai_challenge_2024_nii/xray_10case/mask0817"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".tif"):
            print("*********************************Processing {}**********************************".format(filename))
            input_path = os.path.join(input_dir, filename)
            inference_one_image(input_path, output_dir)
