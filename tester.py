# import sys
# import h5py


# path=r'C:\Storage\Lavinda\Work\iabg\pupillabs\pupilLabs\export\2025-05-07-10-36-27_1\2025-05-07-10-36-27_1.hdf5'


# with h5py.File(path, "r") as h5f:
#     keys = h5f.keys()

#     print(keys)

#     keys=h5f['gaze'].keys()
#     print(keys)

import h5py
import numpy as np

hdf5_file_path = r'C:\Storage\Lavinda\Work\iabg\pupillabs\pupilLabs\export\2025-05-07-10-36-27_1\2025-05-07-10-36-27_1.hdf5'

with h5py.File(hdf5_file_path, "r") as f:
    # list all keys (e.g., 'axis0', 'axis1', etc.)
    keys = list(f.keys())
    print("Keys in file:", keys)
    
    # Create a dictionary with each key mapped to its NumPy array
    data = {}
    for key in keys:
        dataset = f[key]
        data[key] = np.array(dataset)
        print(f"Loaded {key} with shape {data[key].shape}")

# Now you have each dataset as a NumPy array in the 'data' dictionary.
# For example, you can access the "axis0" data as:
axis0_array = data.get("blinks")
print(axis0_array)