import numpy as np
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


def refine_labels(label1, label2):
    """
    Refine label2 based on label1 by adjusting foreground and background regions.

    Parameters:
    label1 (np.ndarray): The reference label.
    label2 (np.ndarray): The label to be refined.

    Returns:
    np.ndarray: Refined label.
    """
    fixed_label2 = label2.copy()

    # Regions that are background in label1 but foreground in label2
    bg_to_fg_mask = (label1 == 0) & (label2 > 0)
    fixed_label2[bg_to_fg_mask] = 0

    return fixed_label2


