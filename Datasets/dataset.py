import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import glob
import os

import torch.utils.data

from utils.utils import find_target_dim

def create_dataloader(config, dir, class_names, bs, FC=False, shuffle=True, drop_last=False):
    """
    Creates a dataloader for the Spectroscopy dataset.
    :param dir: Directory containing .npy files
    :param bs: Batch size
    :param shuffle: Whether to shuffle the data
    :param drop_last: Whether to drop the last incomplete batch
    :param inference_mode: If True, returns dataloader for inference
    """
    dataset = SpectroscopyDataset(config, dir, class_names, FC=FC)
    return torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, drop_last=drop_last), dataset.config

class SpectroscopyDataset(Dataset):

    def __init__(self, config, dir, class_names, FC=False, raw=False):
        """
        Dataset for Spectroscopy data.
        :param dir: Directory containing .npy files
        :param FC: If True, keeps data in fully connected format
        :param raw: If True, returns raw data
        """
        self.X, self.Y = self.read_data(dir, class_names)
        self.target_len, self.config = find_target_dim(self.X.shape[1], config)
        self.FC = FC
        self.raw = raw

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        X = torch.tensor(self.X[index, :])

        if X.shape[0] != self.target_len:
            X = torch.nn.functional.interpolate(
                X.unsqueeze(0).unsqueeze(0), size=self.target_len, mode='linear', align_corners=False
            ).squeeze()

        if not self.FC:
            X = X.unsqueeze(dim=0)  # Add channel dimension for CNN models
        y = torch.tensor(self.Y[index])
        return X.float(), y.long()

    def read_data(self, dir, class_names):
        X, Y = None, None
        class_label = 0  # Starting label for the first class

        for class_name in class_names:
            file = os.path.join(dir, class_name + ".npy")
            data = np.load(file)
            print(f'read {file} and shape is: {data.shape}')

            if X is None:
                X = data
                Y = np.full(data.shape[0], class_label, dtype=int)
            else:
                X = np.concatenate([X, data])
                Y = np.concatenate([Y, np.full(data.shape[0], class_label, dtype=int)])

            class_label += 1  # Increment the class label for the next file

        return X, Y


class InferenceDataset(Dataset):
    """
    Dataset class for inference, handling only features (X).
    """

    def __init__(self, dir, FC=False, raw=False):
        """
        :param dir: Directory containing .npy files
        :param FC: If True, keeps data in fully connected format
        :param raw: If True, returns raw data
        """
        self.X = self.read_data(dir)
        self.FC = FC
        self.raw = raw

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.FC:
            X = torch.tensor(self.X[index, :])
        else:
            X = torch.tensor(self.X[index, :]).unsqueeze(dim=0)
        return X.float()

    def read_data(self, dir):
        """
        Reads data from directory containing .npy files.
        """
        X = None

        for file in glob.glob(os.path.join(dir, "*.npy")):
            data = np.load(file)
            print(f"Read {file} and shape is: {data.shape}")

            if X is None:
                X = data
            else:
                X = np.concatenate([X, data])

        return X


def create_inference_dataloader(dir, bs, FC=False, shuffle=False, drop_last=False):
    """
    Creates a dataloader for inference.
    :param dir: Directory containing .npy files
    :param bs: Batch size
    :param FC: If True, keeps data in fully connected format
    :param shuffle: Whether to shuffle the data
    :param drop_last: Whether to drop the last incomplete batch
    """
    dataset = InferenceDataset(dir, FC=FC)
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, drop_last=drop_last)

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

def create_dataloader_groups(
    path="/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/multispectral/Leone/separate_sample/v1/spectra_dataset_v1_1.npz",
    x_key="X",
    y_key="y_str",
    groups_key="groups",
    wavelengths_key="wavelengths",
    ids_key="ids",
    classes_key="classes",
    batch_size=256,
    val_size=0.2,
    normalize=True,
    use_weighted_sampler=True,
    random_state=42,
    num_workers=0,      # set >0 for speed if you're outside notebooks
    pin_memory=True,    # keep True if training on GPU
    ):
    X, y, groups, feature_names, meta = load_npz(
    path=path,
    x_key=x_key,
    y_key=y_key,
    groups_key=groups_key,
    wavelengths_key=wavelengths_key,
    ids_key=ids_key,
    classes_key=classes_key,
    )
    X, y, groups = filter_labels(X, y, groups, keep_labels=["PE", "PET", "PS", "PP"])

    if len(np.unique(y)) < 2:
        raise ValueError("After keep_labels filtering, at least two classes are required.")
    
    train_loader, val_loader, meta = build_loaders_from_numpy(
        X, y, groups,
        batch_size=batch_size,
        val_size=val_size,
        normalize=normalize,
        use_weighted_sampler=use_weighted_sampler,
        random_state=random_state,
        num_workers=num_workers,      # set >0 for speed if you're outside notebooks
        pin_memory=pin_memory,    # keep True if training on GPU
    )
    return train_loader, val_loader, meta

def _format_feature_names_from_wavelengths(wavelengths: np.ndarray) -> List[str]:
    names = []
    for w in wavelengths:
        try:
            f = float(w)
            names.append(str(int(f)) if abs(f - int(f)) < 1e-9 else str(f))
        except Exception:
            names.append(str(w))
    return names

def load_npz(
    path: str,
    x_key: str = "X",
    y_key: str = "y_str",
    groups_key: str = "groups",
    wavelengths_key: str = "wavelengths",
    ids_key: str = "ids",
    classes_key: str = "classes",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Loads NPZ with expected arrays. Returns:
      X (float32), y (array), groups (array), feature_names (List[str]), meta (dict with ids/classes/wavelengths)
    """
    d = np.load(path, allow_pickle=True)
    if x_key not in d or y_key not in d or groups_key not in d:
        raise KeyError(f"NPZ must contain '{x_key}', '{y_key}', '{groups_key}' (got keys: {list(d.keys())})")

    X = np.asarray(d[x_key]).astype(np.float32)
    y = np.asarray(d[y_key])
    groups = np.asarray(d[groups_key])

    # feature names
    if wavelengths_key in d:
        wavelengths = np.asarray(d[wavelengths_key])
        feature_names = _format_feature_names_from_wavelengths(wavelengths)
    else:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        wavelengths = None

    meta = {
        "ids": np.asarray(d[ids_key]) if ids_key in d else None,
        "classes": np.asarray(d[classes_key]) if classes_key in d else None,
        "wavelengths": wavelengths,
        "npz_keys": list(d.keys()),
    }
    return X, y, groups, feature_names, meta

# ------------------------------
# Optional label filtering (like your example)
# ------------------------------
def filter_labels(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, keep_labels: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not keep_labels:
        return X, y, groups
    keep = set(keep_labels)
    mask = np.array([lbl in keep for lbl in y])
    return X[mask], y[mask], groups[mask]


# ----- Dataset -----
class SpectraDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """
        X: float32 numpy array, shape (N, D)
        y: int64/np.int_ numpy array, shape (N,)
        groups: array-like group ids (kept for reference)
        """
        assert X.shape[0] == y.shape[0] == groups.shape[0]
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.groups = groups  # not used in __getitem__, but handy to keep

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return tensors ready for an MLP (no sequence dimension)
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

# ----- Utilities -----
def encode_labels(y_str: np.ndarray):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    label2id = {c: i for i, c in enumerate(le.classes_)}
    id2label = {i: c for i, c in enumerate(le.classes_)}
    return y_enc, le, label2id, id2label

def group_train_val_split(X, y, groups, val_size=0.2, random_state=42):
    """Disjoint groups between train/val."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(X, y, groups))
    return train_idx, val_idx

def make_weighted_sampler(y_train: np.ndarray):
    """Weighted sampler (inverse frequency) for class imbalance."""
    counts = np.bincount(y_train)
    inv_freq = 1.0 / np.maximum(counts, 1)
    sample_weights = inv_freq[y_train]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )

def compute_class_weights_for_loss(y_train: np.ndarray):
    """Balanced weights for nn.CrossEntropyLoss(weight=...)."""
    n_classes = int(np.max(y_train)) + 1
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    # classic "balanced" weights: N / (C * count_c)
    weights = (len(y_train) / (n_classes * np.maximum(counts, 1.0))).astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32)

# ----- Main builder -----
def build_loaders_from_numpy(
    X: np.ndarray,
    y_str: np.ndarray,
    groups: np.ndarray,
    batch_size: int = 256,
    val_size: float = 0.2,
    normalize: bool = True,
    use_weighted_sampler: bool = True,
    random_state: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Returns:
      train_loader, val_loader, meta where meta contains:
        - label_encoder, id2label, label2id
        - scaler (or None)
        - class_weights (torch.FloatTensor for loss)
        - indices: train_idx, val_idx
    """
    # 1) Encode labels
    y, le, label2id, id2label = encode_labels(y_str)

    # 2) Group-aware split
    train_idx, val_idx = group_train_val_split(X, y, groups, val_size=val_size, random_state=random_state)

    # 3) Standardize using train statistics only
    scaler = None
    if normalize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X[train_idx])
        X_train = scaler.transform(X[train_idx]).astype(np.float32)
        X_val   = scaler.transform(X[val_idx]).astype(np.float32)
    else:
        X_train = X[train_idx].astype(np.float32, copy=False)
        X_val   = X[val_idx].astype(np.float32, copy=False)

    y_train = y[train_idx]
    y_val   = y[val_idx]
    g_train = groups[train_idx]
    g_val   = groups[val_idx]

    # 4) Build datasets
    ds_train = SpectraDataset(X_train, y_train, g_train)
    ds_val   = SpectraDataset(X_val,   y_val,   g_val)

    # 5) Optional weighted sampler (train only)
    sampler = make_weighted_sampler(y_train) if use_weighted_sampler else None

    # 6) DataLoaders
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=min(batch_size * 2, max(1, len(ds_val))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # 7) Class weights for loss (useful even if you don't use the sampler)
    class_weights = compute_class_weights_for_loss(y_train)

    meta = {
        "label_encoder": le,
        "label2id": label2id,
        "id2label": id2label,
        "scaler": scaler,
        "class_weights": class_weights,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }
    return train_loader, val_loader, meta

# ----- Example usage with YOUR arrays (already loaded & filtered) -----
# X, y, groups are the numpy arrays you printed above

# ------------------------------
# Grouped, stratified folds with validation (new)
# ------------------------------
def grouped_stratified_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    val_size: float = 0.2,
    seed: int = 42,
) -> List[Dict[str, List[int]]]:
    """
    Returns list of dicts with keys: fold, train_idx, val_idx, test_idx
    - Outer split: grouped (stratified if possible).
    - Inner split: from train+val, carve out val (grouped, stratified if possible).
    - Reproducible with 'seed'; val split uses (seed + fold).
    """
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        has_sgkf = True
    except Exception:
        has_sgkf = False

    y = np.asarray(y)
    groups = np.asarray(groups)
    n = y.shape[0]

    # Outer iterator
    def _outer_iter():
        if has_sgkf:
            try:
                sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                yield from sgkf.split(np.zeros(n), y, groups)
                return
            except ValueError:
                pass
        gkf = GroupKFold(n_splits=n_splits)
        yield from gkf.split(np.zeros(n), y, groups)

    folds: List[Dict[str, List[int]]] = []
    for fold_id, (trval_idx, te_idx) in enumerate(_outer_iter()):
        y_trval, g_trval = y[trval_idx], groups[trval_idx]

        # Inner val split
        if has_sgkf:
            try:
                # derive a number of splits so that test_size≈val_size
                n_val = max(2, int(round(1.0 / max(1e-6, val_size))))
                inner = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed + fold_id)
                tr_i, va_i = next(inner.split(np.zeros_like(y_trval), y_trval, g_trval))
            except ValueError:
                inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + fold_id)
                tr_i, va_i = next(inner.split(np.zeros_like(y_trval), y_trval, g_trval))
        else:
            inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + fold_id)
            tr_i, va_i = next(inner.split(np.zeros_like(y_trval), y_trval, g_trval))

        tr_idx = trval_idx[tr_i]
        va_idx = trval_idx[va_i]

        folds.append({
            "fold": int(fold_id),
            "train_idx": tr_idx.tolist(),
            "val_idx": va_idx.tolist(),
            "test_idx": te_idx.tolist(),
        })

    return folds


# if __name__=="__main__":


#     print("Classes:", meta["label_encoder"].classes_)
#     print("Label → id:", meta["label2id"])
#     print("Train batches:", len(train_loader), " Val batches:", len(val_loader))

#     # ----- Example: wiring into a simple training loop -----
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     from model import FCNet
#     model = FCNet(input_size=X.shape[1], num_classes=len(meta["label_encoder"].classes_)).to(device)

#     criterion = torch.nn.CrossEntropyLoss(weight=meta["class_weights"].to(device))
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         logits = model(xb)
#         loss = criterion(logits, yb)
#         loss.backward()
#         optimizer.step(); optimizer.zero_grad()

if __name__=="__main__":
        dir = ""
        dataset = SpectroscopyDataset(dir, FC=True)
        dataloader = DataLoader(dataset, batch_size=2,shuffle=True)
        for data, labels, data_num in dataloader:
            print(data.dtype)
            print(labels.dtype)
