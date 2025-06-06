# Pupil Labs Data Extraction

This project is to extract data from pupil labs recording file. The extraction is available in 2 forms, which is a semi-colon seperated file and an hdf5 file.

The input parameters is in config.json where user can and should input the local path of the computer. 

## HDF5 file

Pandas data structure is saved in this file format to allow easy access to the results withut recalculating values. an example on how to read the file is available in

`examples/hdf5_reader.py`

## Doxygen documentation

this project uses the library doxygen, installing link can be found [here](https://www.doxygen.nl/download.html), full documentation can be found here [doxygen_docu](https://www.doxygen.nl/manual/index.html). 

in this repository, a doxygen file is already initialize. To regenerate documentation, run `doxygen Doxyfile`

