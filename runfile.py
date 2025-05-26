import lib.getData as getData
from pathlib import Path
import h5py
import pandas as pd
import numpy as np

recording_file=r'C:\Storage\Lavinda\Work\iabg\pupillabs\pupilLabs\testDataset'
recording_number=r'2025-05-07-10-36-27_1'
export_path=r'C:\Storage\Lavinda\Work\iabg\pupillabs\pupilLabs\export'


getData.export(csv=True, hdf5=True, recording_file=recording_file, recording_number=recording_number, export_path=export_path)




