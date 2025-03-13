import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split


def split_npy_file(input_dir='./input_npy', output_dirs={'train':'./train/','val':'./val/','test':'./test/'}):
# Ensure output directories exist
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Train, validation, test split ratios
    train_ratio = 0.7
    val_ratio = 0.15

    # List all .npy files in the input directory
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for npy_file in npy_files:
        # Load the .npy file
        file_path = os.path.join(input_dir, npy_file)
        data = np.load(file_path)

        # Split data into train, val, and test
        train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        # Save the splits to the respective directories
        np.save(os.path.join(output_dirs['train'], npy_file), train_data)
        np.save(os.path.join(output_dirs['val'], npy_file), val_data)
        np.save(os.path.join(output_dirs['test'], npy_file), test_data)

    print("Splitting and saving completed.")

if __name__=="__main__":
    # Define input and output directories
    input_dir = '/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Leone/all_labeled/'  # Directory containing the .npy files
    output_dirs = {
        'train': '/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Leone/all_labeled/train/',
        'val': '/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Leone/all_labeled/val/',
        'test': '/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Leone/all_labeled/test/'
    }
    split_npy_file(input_dir, output_dirs)