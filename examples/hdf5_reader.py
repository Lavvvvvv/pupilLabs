'''
example code to read a hdf5 file and print the contents of the 'gaze' key
'''
import h5py
import numpy as np
import pandas as pd

hdf5_file_path = r'C:\Storage\Lavinda\Work\iabg\pupillabs\pupilLabs\export\2025-05-07-10-36-27_1\2025-05-07-10-36-27_1.hdf5'

df = pd.read_hdf(hdf5_file_path, key="gaze")
print(df)