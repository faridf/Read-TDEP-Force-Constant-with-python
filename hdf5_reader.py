import h5py
import numpy as np

def explore_hdf5_structure(filename):
    """
    Explore and print the structure of an HDF5 file.
    """
    def print_structure(name, obj):
        """Callback to print the name and type of HDF5 objects"""
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}")
            print(f"  Shape: {obj.shape}")
            print(f"  Type: {obj.dtype}")
            print()
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}/")
            print()

    with h5py.File(filename, 'r') as f:
        print("HDF5 File Structure:")
        print("===================")
        f.visititems(print_structure)

def read_dispersion_data(filename):
    """
    Read eigenvalues and eigenvectors from the HDF5 file.
    Returns a dictionary containing the data.
    """
    data = {}
    
    with h5py.File(filename, 'r') as f:
        # Print all available keys at root level
        print("Available root keys:", list(f.keys()))
        
        # Try to read the most common keys for dispersion relations
        try:
            if 'eigenvalues' in f:
                data['eigenvalues'] = f['eigenvalues'][:]
            if 'eigenvectors' in f:
                data['eigenvectors'] = f['eigenvectors'][:]
            if 'qpoints' in f:
                data['qpoints'] = f['qpoints'][:]
            if 'frequencies' in f:
                data['frequencies'] = f['frequencies'][:]
                
            # Add any additional metadata
            for key in f.keys():
                if key not in data:
                    try:
                        data[key] = f[key][:]
                    except:
                        print(f"Could not read dataset: {key}")
                        
        except Exception as e:
            print(f"Error reading data: {str(e)}")
    
    return data

def main():
    filename = 'outfile.dispersion_relations.hdf5'
    
    # First, explore the file structure
    print("Exploring file structure...")
    explore_hdf5_structure(filename)
    
    # Then read the data
    print("\nReading dispersion data...")
    data = read_dispersion_data(filename)
    
    # Print summary of what was found
    print("\nData Summary:")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Type: {value.dtype}")
            if value.size > 0:
                print(f"  Min: {value.min()}")
                print(f"  Max: {value.max()}")
            print()

if __name__ == "__main__":
    main()