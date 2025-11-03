import os
import numpy as np
from pathlib import Path

# Paths and constants
base_dir = Path("/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/sg/1/")
output_dirs = ["/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/sg/2",
               "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/sg/3",
               "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/sg/4",
               "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/sg/5"]
splits = ["train", "test", "val"]

# Get class files (assume same names in train/test/val)
class_files = sorted((base_dir / "train").glob("*.npy"))

# Determine split ratios from original dataset
split_sizes = {}
for file in class_files:
    cls = file.name
    train_size = np.load(base_dir / "train" / cls).shape[0]
    test_size = np.load(base_dir / "test" / cls).shape[0]
    val_size = np.load(base_dir / "val" / cls).shape[0]
    total = train_size + test_size + val_size
    split_sizes[cls] = {
        "train": train_size / total,
        "test": test_size / total,
        "val": val_size / total
    }

# Repeat for each new dataset version
for out_dir in output_dirs:
    for split in splits:
        Path(out_dir, split).mkdir(parents=True, exist_ok=True)
    
    for file in class_files:
        cls = file.name

        # Combine all data for this class
        data_train = np.load(base_dir / "train" / cls)
        data_test = np.load(base_dir / "test" / cls)
        data_val = np.load(base_dir / "val" / cls)
        all_data = np.vstack([data_train, data_test, data_val])

        # Shuffle all rows
        np.random.shuffle(all_data)

        # Split based on original ratios
        n = all_data.shape[0]
        n_train = int(split_sizes[cls]["train"] * n)
        n_test = int(split_sizes[cls]["test"] * n)
        
        train_data = all_data[:n_train]
        test_data = all_data[n_train:n_train + n_test]
        val_data = all_data[n_train + n_test:]

        # Save to output directories
        np.save(Path(out_dir, "train", cls), train_data)
        np.save(Path(out_dir, "test", cls), test_data)
        np.save(Path(out_dir, "val", cls), val_data)
