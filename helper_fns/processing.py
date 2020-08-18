



import numpy as np


def scale_and_remove_na(np_array):
    
    # Getting and storing parameters
    min_val = np.nanmin(np_array)
    max_val = np.nanmax(np_array)
    # Scaling between 0 and 1
    np_array_scaled = (np_array - min_val) / (max_val - min_val)
    # Replacing nan with 0
    np_array_scaled_no_nans = np.nan_to_num(np_array_scaled)
    
    return(np_array_scaled_no_nans)