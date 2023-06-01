import numpy as np
from skimage.measure import label as labeler
from skimage.measure import regionprops


def get_number_of_pixels_in_hectare(meter_resolution: float) -> float:
    return 10_000 / meter_resolution**2


def get_contiguous_areas_of_class_with_maximum_size(X_data: np.ndarray,
                                                    class_label: int | list[int],
                                                    max_contiguous_pixel_area: int) -> np.ndarray:
    if isinstance(class_label, int):
        class_labels = [class_label]
    elif isinstance(class_label, list):
        class_labels = class_label
    else:
        raise TypeError('Class_label must be int or list of ints')

    if X_data.dtype not in ['uint8', 'int32', 'int64']:
        raise ValueError('Please recast array as integer array')

    X_binary = np.isin(X_data, class_labels).astype(np.uint8)
    label_arr = labeler(X_binary,
                        connectivity=1,
                        background=0)
    props = regionprops(label_arr)

    labels_to_exclude = [(k+1) for (k, prop) in enumerate(props) if prop.area <= max_contiguous_pixel_area]
    size_mask = np.isin(label_arr, labels_to_exclude)

    return size_mask
