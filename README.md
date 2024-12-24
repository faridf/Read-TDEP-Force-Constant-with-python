# Phonon Dispersion Data Analysis with TDEP Force Constant

This repository contains Python scripts for analyzing phonon dispersion relations data generated by the [TDEP code](https://tdep-developers.github.io/tdep/). Specifically, it processes the HDF5 output file (`outfile.dispersion_relations.hdf5`) produced by TDEP's phonon dispersion relations calculations.

## Overview

The scripts provide tools for:
- Reading and processing TDEP's HDF5 phonon dispersion data
- Extracting eigenvalues (frequencies) and eigenvectors
- Analyzing group velocities and site projections
- Visualizing phonon dispersion relations
- Converting data to more accessible formats (numpy)

## Requirements

```
numpy
h5py
matplotlib
```

You can install the requirements using:
```bash
pip install numpy h5py matplotlib
```

## File Structure

- `phonon_data_handler.py`: Main script for loading and analyzing phonon data
- `hdf5_reader.py`: Utility script for exploring HDF5 file structure
- `phonon_dispersion.png`: Generated dispersion relation plot
- `phonon_data.npz`: Numpy format data file (generated by the script)

## Data Structure

The HDF5 file from TDEP contains:
- `frequencies` (shape: [400, 645]): Phonon frequencies at each q-point
- `eigenvectors_re` and `eigenvectors_im` (shape: [400, 645, 645]): Real and imaginary parts of eigenvectors
- `group_velocities` (shape: [400, 645, 3]): Group velocities for each mode at each q-point
- `q_values` (shape: [400]): Points along the chosen path in the Brillouin zone
- `q_vector` (shape: [400, 3]): Q-point vectors
- `q_ticks` (shape: [5]): Special points along the path
- `site_projection_per_mode` (shape: [400, 645, 215]): Mode projections onto atomic sites

## Usage

### Basic Usage

```python
from phonon_data_handler import PhononData

# Load the data
phonon_data = PhononData('outfile.dispersion_relations.hdf5')

# Get frequencies at a specific q-point (e.g., Γ point)
frequencies_gamma = phonon_data.get_frequencies_at_q(0)

# Plot dispersion relations
phonon_data.plot_dispersion()

# Save data in numpy format
phonon_data.save_numpy('phonon_data.npz')
```

### Exploring HDF5 Structure

To examine the structure of your HDF5 file:
```python
python hdf5_reader.py
```

This will print detailed information about the datasets in your file.

## Class Methods

The `PhononData` class provides several methods:

- `load_data(filename)`: Load data from HDF5 file
- `get_frequencies_at_q(q_idx)`: Get frequencies at specific q-point
- `get_eigenvectors_at_q(q_idx)`: Get eigenvectors at specific q-point
- `get_group_velocities_at_q(q_idx)`: Get group velocities at specific q-point
- `plot_dispersion(ylim=None, save_path=None)`: Plot phonon dispersion relations
- `save_numpy(filename)`: Save data in numpy format

## Connection to TDEP

This code is designed to work with output from the Temperature Dependent Effective Potential (TDEP) method. TDEP is a powerful tool for calculating temperature-dependent interatomic force constants and phonon properties. The HDF5 file analyzed by these scripts is produced by TDEP's `phonon_dispersion_relations` program.

For more information about TDEP and its phonon dispersion calculations, visit:
- [TDEP Documentation](https://tdep-developers.github.io/tdep/)
- [Phonon Dispersion Relations in TDEP](https://tdep-developers.github.io/tdep/program/phonon_dispersion_relations/#__tabbed_1_1)

## Output Files

The script generates several output files:
1. `phonon_dispersion.png`: Visual representation of the phonon dispersion relations
2. `phonon_data.npz`: Numpy archive containing the processed data

## Note on Units

The frequencies in the output are typically in THz or meV (check TDEP documentation for specific units used in your calculation). The q-points are given in reciprocal lattice units.

## Contributing

Feel free to submit issues and enhancement requests!


## Citation

If you use this code in your research, please cite:
1. The TDEP method: [Hellman et al., Phys. Rev. B 87, 104111 (2013)](https://doi.org/10.1103/PhysRevB.87.104111)
