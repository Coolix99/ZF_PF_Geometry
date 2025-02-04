import numpy as np
import os
from zf_pf_geometry.image_operations import get_Image

def load_tif_image(folder):
    """
    Loads a single `.tif` image from the specified folder.

    Args:
        folder (str): Path to the folder containing `.tif` files.

    Returns:
        np.ndarray: Loaded image.
    """
    img_list = [item for item in os.listdir(folder) if item.endswith('.tif')]
    if len(img_list) != 1:
        raise ValueError(f"Expected 1 .tif file, found {len(img_list)} in {folder}")
    return get_Image(os.path.join(folder, img_list[0]))

def check_array_type(array: np.ndarray) -> str:
    """
    Check whether a NumPy array has a floating-point type or discrete type.

    Args:
        array (np.ndarray): The array to check.

    Returns:
        str: "float" if the array is floating-point,
             "discrete" if the array is bool/int/uint,
             "unknown" otherwise.
    """

    if np.issubdtype(array.dtype, np.floating):
        return "float"
    elif np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        return "discrete"
    else:
        return "unknown"
    
def make_path(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)