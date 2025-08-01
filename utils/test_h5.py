import h5py

file_path = '/Users/darklord/Research/VAR/code/var-mri/brats2020-kaggle/archive/BraTS2020_training_data/content/data/volume_1_slice_0.h5'

with h5py.File(file_path, 'r') as f:
    print("Keys:", list(f.keys()))
    for key in f.keys():
        print(f"{key}: shape = {f[key].shape}, dtype = {f[key].dtype}")
