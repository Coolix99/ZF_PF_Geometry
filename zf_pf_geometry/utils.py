import numpy as np
import os

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