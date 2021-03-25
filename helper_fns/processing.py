



import numpy as np


def scale_and_remove_na(np_array, parameter_storage, index):

    print('Functionality deprecation: scale_and_remove_na no longer removes \
           na values because they should now be dealed with in other stages')
    
    # Getting parameters for transformation
    min_val = np.nanmin(np_array)
    max_val = np.nanmax(np_array)
    # Storing those parameters for inversion transformation
    parameter_storage[0, index] = min_val
    parameter_storage[1, index] = max_val
    # Scaling between 0 and 1
    np_array_scaled = (np_array - min_val) / (max_val - min_val)
    # Replacing nan with 0
    #np_array_scaled_no_nans = np.nan_to_num(np_array_scaled)
    
    return(np_array_scaled, parameter_storage)