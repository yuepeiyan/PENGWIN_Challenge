import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label, find_objects


def split_connected_components(labels, label_value, offset, min_volume=400, top_n=6):
    """
    Split the region with the specified label_value into multiple connected components and reassign labels.

    Parameters:
    labels (np.ndarray): Input label array
    label_value (int): The label value to split
    offset (int): Offset used to generate new label values
    min_volume (int): Minimum volume to retain connected components
    top_n (int): Retain the top-n connected components by volume

    Returns:
    np.ndarray: Relabeled array
    """
    # Get a binary mask where the label is equal to label_value
    binary_mask = (labels == label_value)

    structure = np.array([[[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]],
                          [[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]],
                          [[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]], dtype=int)

    # Use scipy.ndimage.label to mark connected components
    labeled_array, num_features = label(binary_mask, structure=structure)

    # Create new_labels as a copy of the input labels
    new_labels = labels.copy()

    # Get the volume of all connected components
    volumes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]

    # Get indices of the top-n connected components by volume
    top_n_indices = np.argsort(volumes)[-top_n:][::-1]
    top_n_volumes_labels = [(volumes[i], i + 1) for i in top_n_indices]  # Note that component indices start from 1

    # Iterate through all connected components in descending order of volume and reassign labels to avoid conflicts
    current_label = offset
    for volume, i in top_n_volumes_labels:
        region_mask = (labeled_array == i)
        if volume >= min_volume:
            new_labels[region_mask] = current_label
            current_label += 1
        else:
            new_labels[region_mask] = 0

    return new_labels


def remove_small_connected_components(prediction, min_volume, label_values):
    """
    Remove small connected components and set them as background.

    Parameters:
    prediction (np.ndarray): Model output predictions
    min_volume (int): Minimum volume to retain connected components
    label_values (list): List of label values to process

    Returns:
    np.ndarray: Processed prediction array
    """
    new_prediction = prediction.copy()

    # Define the connectivity structure for identifying connected components
    structure = np.array([[[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]],
                          [[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]],
                          [[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]], dtype=int)

    for index, label_value in enumerate(label_values):
        print(f"Processing label {label_value}:")
        # Get binary mask for the specified label
        binary_mask = (prediction == label_value)
        minimum = min_volume[index]

        labeled_array, num_features = label(binary_mask, structure=structure)

        # Get slices of each connected component
        slices = find_objects(labeled_array)

        retained_sizes = []
        removed_sizes = []

        # Iterate through each connected component and remove those smaller than the minimum volume
        for i, slice_ in enumerate(slices):
            region_size = np.sum(labeled_array[slice_] == (i + 1))
            if region_size <= minimum:
                removed_sizes.append(region_size)
                new_prediction[labeled_array == (i + 1)] = 0
            else:
                retained_sizes.append(region_size)

        # Print the sizes of retained and removed regions
        if retained_sizes:
            print(f"  Retained regions sizes: {retained_sizes}")
        if removed_sizes:
            print(f"  Removed regions sizes: {removed_sizes}")

    return new_prediction



def calculate_iou(label1, label2):
    intersection = np.logical_and(label1, label2).sum()
    union = np.logical_or(label1, label2).sum()
    return intersection / union


def relabel_connected_components(segmentation):
    """
    Handle partial confusion between left and right hip bones in stage0 segmentation.

    Parameters:
    segmentation (np.ndarray): The segmentation labels from stage0.

    Returns:
    np.ndarray: Relabeled segmentation with confusion resolved.
    """
    # Detect connected components for labels 2 and 3
    label_2, num_features_2 = ndimage.label(segmentation == 2)
    label_3, num_features_3 = ndimage.label(segmentation == 3)

    # Create arrays to store the sizes of the connected components
    size_2 = np.bincount(label_2.ravel())
    size_3 = np.bincount(label_3.ravel())

    # Initialize a new segmentation array for relabeling
    new_segmentation = np.copy(segmentation)

    # Create a structural element to detect boundaries, suitable for 3D
    struct = ndimage.generate_binary_structure(3, 1)

    # Iterate over the connected components for label 2
    for label in range(1, num_features_2 + 1):
        current_region = (label_2 == label)
        neighbors = ndimage.binary_dilation(current_region, structure=struct) & (segmentation == 3)

        if neighbors.any():
            touching_labels_3 = np.unique(label_3[neighbors])
            for lbl_3 in touching_labels_3:
                if lbl_3 > 0:
                    if 5 * size_2[label] < size_3[lbl_3]:
                        print(f"Change: class_2 (size: {size_2[label]}) -> class_3 (size: {size_3[lbl_3]})")
                        new_segmentation[current_region] = 3
                    elif 5 * size_3[lbl_3] < size_2[label]:
                        print(f"Change: class_3 (size: {size_3[lbl_3]}) -> class_2 (size: {size_2[label]})")
                        new_segmentation[label_3 == lbl_3] = 2

    return new_segmentation


def refine_labels(label1, label2, threshold=0.99):
    """
    Refine label2 based on reference from label1 if IoU is below the threshold.

    Parameters:
    label1 (np.ndarray): The reference label.
    label2 (np.ndarray): The label to be refined.
    threshold (float): IoU threshold for refinement. Default is 0.99.

    Returns:
    np.ndarray: Refined label.
    """
    iou = calculate_iou(label1 > 0, label2 > 0)  # Calculate IoU considering only foreground and background
    if iou >= threshold:
        return label2

    print('Refining label...')
    fixed_label2 = label2.copy()

    # Label the connected components in label2
    structure = np.array([[[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]],
                          [[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]],
                          [[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]], dtype=int)
    labeled_a, num_features_a = label(label2, structure=structure)

    # Iterate over the connected components in label2
    for component_id in range(1, num_features_a + 1):
        component_mask = (labeled_a == component_id)
        if not np.any(component_mask & (label1 > 0)):
            # If there is no intersection with label1, set the component to background
            fixed_label2[component_mask] = 0

    # Foreground areas in label1 that are background in label2
    fg_to_bg_mask = (label1 > 0) & (label2 == 0)

    # Find the nearest foreground pixel label
    if fg_to_bg_mask.any():
        distance, indices = distance_transform_edt(fixed_label2 == 0, return_indices=True)
        nearest_foreground = label2[tuple(indices)]
        fixed_label2[fg_to_bg_mask] = nearest_foreground[fg_to_bg_mask]

    return fixed_label2


def process_final_label(segmentation):
    """
    Process the final segmentation labels by refining the connected components of specific labels.

    Parameters:
    segmentation (np.ndarray): The final segmentation labels.

    Returns:
    np.ndarray: Refined segmentation with certain connected components removed or relabeled.
    """
    # Initialize a new segmentation array for relabeling
    new_segmentation = np.copy(segmentation)

    # Mask out sacrum labels (set to background)
    segmentation = np.where((segmentation >= 1) & (segmentation <= 10), 0, segmentation)

    # Detect connected components for labels 11 and 21
    label_11, num_features_11 = ndimage.label(segmentation == 11)
    label_21, num_features_21 = ndimage.label(segmentation == 21)

    # Calculate the size of each connected component
    size_11 = np.bincount(label_11.ravel())
    size_21 = np.bincount(label_21.ravel())

    # Find the index of the largest connected component for labels 11 and 21
    largest_label_11_index = np.argmax(size_11[1:]) + 1  # Skip index 0 (background)
    largest_label_21_index = np.argmax(size_21[1:]) + 1

    assert num_features_11 > 0 and num_features_21 > 0, "label 11 and label 21 have no connected components!!"

    # Remove the largest connected components from label_11 and label_21 (mark them as background)
    label_11[label_11 == largest_label_11_index] = 0  # Mark largest connected component as background
    num_features_11 -= 1  # Update the number of connected components

    label_21[label_21 == largest_label_21_index] = 0  # Mark largest connected component as background
    num_features_21 -= 1  # Update the number of connected components

    # Create a structural element for boundary detection, suitable for 3D
    struct = ndimage.generate_binary_structure(3, 1)

    # Define a function to process connected components for a given label
    def process_label(label, segment_label, num_features):
        if num_features < 1:
            return  # Do not process if no connected components remain after removing the largest one

        for lbl in range(1, num_features + 1):
            current_region = (label == lbl)
            neighbors = ndimage.binary_dilation(current_region, structure=struct) & (segmentation != segment_label)

            if neighbors.any():
                # Find all touching labels, excluding background
                touching_labels = np.unique(segmentation[neighbors])
                touching_labels = touching_labels[touching_labels != 0]  # Exclude background
                touching_labels = touching_labels[touching_labels != segment_label]  # Exclude current label

                if touching_labels.size > 0:
                    # Calculate the volume of each touching label
                    touching_label_sizes = {label: np.sum(segmentation == label) for label in touching_labels}

                    # Find the touching label with the largest volume
                    max_touching_label = max(touching_label_sizes, key=touching_label_sizes.get)
                    print(f"Changing segment {lbl} from {segment_label} to {max_touching_label}")
                    new_segmentation[current_region] = max_touching_label

    # Process connected components for label 11
    process_label(label_11, 11, num_features_11)

    # Process connected components for label 21
    process_label(label_21, 21, num_features_21)

    return new_segmentation


if __name__ == "__main__":
    labels = sitk.ReadImage(
        "/home/ypy/Code/PENGWIN-example-algorithm-main/PENGWIN-challenge-packages/preliminary-development-phase-ct/stage1_label_after_remove_101_1.nii.gz")
    spacing = labels.GetSpacing()
    direction = labels.GetDirection()
    origin = labels.GetOrigin()
    labels = sitk.GetArrayFromImage(labels)
    # label_value = 2
    # offset = 22
    # new_labels = split_connected_components(labels, label_value, offset)
    # new_labels = np.where(new_labels == 1, 21, new_labels)
    # new_labels = remove_small_connected_components(labels, 20000, [1, 2, 3])
    new_labels = relabel_connected_components(labels)
    save_label = sitk.GetImageFromArray(new_labels.astype(np.int8))
    save_label.SetSpacing(spacing)
    save_label.SetDirection(direction)
    save_label.SetOrigin(origin)
    sitk.WriteImage(save_label, "stage1_label_after.nii.gz", useCompression=True)
