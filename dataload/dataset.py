"""
Dataset (7-Task Version) - FIXED TEST LOADER
"""
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from configs.config import cfg

class H5RamDataset(Dataset):
    def __init__(self, h5_path, split):
        self.h5_path = h5_path
        self.split = split
        try:
            with h5py.File(self.h5_path, 'r') as f:
                # Check if keys exist
                if f'{split}_radar' not in f:
                    print(f"Warning: Split '{split}' not found in {h5_path}")
                    self.radar = []
                    return

                self.radar = f[f'{split}_radar'][:]
                self.ecg = f[f'{split}_ecg'][:][:, :2, :]  # Keep only first 2 leads (I, II)
                self.icg = f[f'{split}_icg'][:]
                self.dicg = f[f'{split}_dicg'][:]
                self.bp = f[f'{split}_bp'][:]
                self.flags = f[f'{split}_flags'][:]
                self.mask = f[f'{split}_mask'][:]
                print(f"Loaded {split}: {len(self.radar)} samples")
        except Exception as e:
            print(f"Error loading {split} split: {e}")
            self.radar = []

    def __len__(self):
        return len(self.radar)

    def __getitem__(self, idx):
        def to_tensor(x):
            t = torch.from_numpy(x).float()
            return torch.nan_to_num(t, nan=0.0)

        return {
            'radar_i': to_tensor(self.radar[idx]),
            'ecg': to_tensor(self.ecg[idx]), 
            'icg': to_tensor(self.icg[idx]),
            'dicg': to_tensor(self.dicg[idx]),
            'bp': to_tensor(self.bp[idx]),
            'flags': torch.from_numpy(self.flags[idx]).float(),
            'mask': to_tensor(self.mask[idx])
        }

def create_patient_wise_splits(config):
    maneuver_name = config.maneuvers_to_load[0][1] 
    h5_path = config.h5_file_pattern.format(m=maneuver_name)
    
    print(f"Loading datasets from: {h5_path}")
    
    # 1. Train
    train_ds = H5RamDataset(h5_path, 'train')
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    # 2. Val
    val_ds = H5RamDataset(h5_path, 'val')
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # 3. Test (This was missing/None before)
    test_ds = H5RamDataset(h5_path, 'test')
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader