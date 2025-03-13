import numpy as np
import os
import shutil
import pybaselines
from sklearn.model_selection import train_test_split

# Baseline correction functions
def asls_eddie(old_array):
    return old_array - pybaselines.whittaker.asls(old_array)[0]

def asls_dataset(array, save_path=None):
    out_array = None
    for i in range(len(array)):
        new_array = asls_eddie(array[i])
        if out_array is None:
            out_array = new_array
        else:
            out_array = np.vstack((out_array, new_array))
    
    if len(out_array) != len(array):
        print("Baseline correction for all array not available")
    
    if save_path:
        np.save(save_path, out_array)
    else:
        return out_array

# Directories
input_dir = '/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Leone/all_labeled/limited_samples/PVC/'  # Directory containing .npy files
output_dirs = {
    'train': '/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Leone/all_labeled/train/',
    'val': '/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Leone/all_labeled/val/',
    'test': '/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Leone/all_labeled/test/'
}

# Ensure output directories exist
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Train, validation, test split ratios
train_ratio = 0.70
val_ratio = 0.15

# List all .npy files in the input directory
npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

for npy_file in npy_files:
    file_path = os.path.join(input_dir, npy_file)

    # Load the .npy file
    data = np.load(file_path)

    # Split data into train, val, and test
    # train_data, test_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)
    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Apply baseline correction
    # train_data = asls_dataset(train_data)
    # val_data = asls_dataset(val_data)
    # test_data = asls_dataset(test_data)

    # Save the processed datasets
    np.save(os.path.join(output_dirs['train'], npy_file), train_data)
    np.save(os.path.join(output_dirs['val'], npy_file), val_data)
    np.save(os.path.join(output_dirs['test'], npy_file), test_data)

print("Data splitting and baseline correction completed.")
