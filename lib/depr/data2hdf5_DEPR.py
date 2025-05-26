import h5py, numpy as np, numbers
import os

"""
This module provides functions to save data to HDF5 files using the h5py library.
Deprecated:  translating to hdf5 is built in to getData
"""


def save_to_hdf5(h5File,dataset_name :str, data):
    """
    type matching data to h5py dataset
    ├── Dict           → subgroup (then recurse)
    ├── list / tuple   → variable‑length dataset
    ├── NumPy array    → dataset
    ├── scalar number  → 0‑D dataset
    └── str            → UTF‑8 dataset
    """
    if isinstance(data, dict):
        save_dict_to_hdf5(h5File, data)

    elif isinstance(data, (list, tuple)):
        h5File.create_dataset(
            "data",
            data=np.asarray(data, dtype=object),
            dtype=h5py.vlen_dtype(np.dtype("O"))
        )

    elif isinstance(data, str):
        dt = h5py.string_dtype(encoding="utf‑8")
        h5File.create_dataset(dataset_name, data=np.array(data, dtype=dt), dtype=dt)

    elif isinstance(data, (np.ndarray, numbers.Number)):
        h5File.create_dataset(dataset_name, data=data)

    else:
        raise TypeError(f"Don't know how to store type {type(data)}")

def save_dict_to_hdf5(h5File, data : dict):
    """
    Recursively write a nested dict (or list/tuple) into an open h5py Group.
    ├── Dict           → subgroup (then recurse)
    ├── list / tuple   → variable‑length dataset
    ├── NumPy array    → dataset
    ├── scalar number  → 0‑D dataset
    └── str            → UTF‑8 dataset
    """
    for key, value in data.items():
        if isinstance(value, dict):
            save_dict_to_hdf5(h5File.require_group(key), value)

        elif isinstance(value, (list, tuple)):
            h5File.create_dataset(
                key,
                data=np.asarray(value, dtype=object),
                dtype=h5py.vlen_dtype(np.dtype("O"))
            )

        elif isinstance(value, str):
            dt = h5py.string_dtype(encoding="utf‑8")
            h5File.create_dataset(key, data=np.array(value, dtype=dt), dtype=dt)

        elif isinstance(value, (np.ndarray, numbers.Number)):
            h5File.create_dataset(key, data=value)

        else:
            raise TypeError(f"Don't know how to store type {type(value)} for key '{key}'")



def create_hdf5(log_data,dataset_name, result_path: str, append: bool = False) -> None:
    '''
    Create an HDF5 file and save the log data to it.
    Parameters
    ----------
    log_data : dict, list, tuple, str, np.ndarray, numbers.Number
        The data to be saved in the HDF5 file.
    result_path : str
        The path where the HDF5 file will be saved.
    append : bool, optional
        If True, the data will be appended to the existing file. 
        If False, the file will be overwritten. Default is False.
    '''
    if append:
        mode = "a"
    else:
        mode = "w"

    if mode == "w":
        # If the file already exists and is not empty, generate a new filename.
        base, ext = os.path.splitext(result_path)
        counter = 0
        # Check if the file exists and is not empty.
        while os.path.exists(result_path):
            counter += 1
            result_path = f"{base}({counter}){ext}"
        mode = "w"

    with h5py.File(result_path, mode) as f:
        save_to_hdf5(f,dataset_name, log_data)
