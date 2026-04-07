"""
Enhanced Dataset Module with LOSO Support
Provides patient-wise data splitting for Leave-One-Subject-Out evaluation.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from configs.config import cfg

class H5RamDataset(Dataset):
    def __init__(self, h5_path, split, patient_ids=None):
        self.h5_path = h5_path
        self.split = split
        self.patient_ids = patient_ids
        
        try:
            with h5py.File(self.h5_path, 'r') as f:
                # Check if keys exist
                if f'{split}_radar' not in f:
                    print(f"Warning: Split '{split}' not found in {h5_path}")
                    self.radar = []
                    return

                # Load all data
                self.radar = f[f'{split}_radar'][:]
                self.ecg = f[f'{split}_ecg'][:]
                self.icg = f[f'{split}_icg'][:]
                self.dicg = f[f'{split}_dicg'][:]
                self.bp = f[f'{split}_bp'][:]
                self.strain = f[f'{split}_strain'][:]
                self.resp = f[f'{split}_resp'][:]
                self.flags = f[f'{split}_flags'][:]
                self.mask = f[f'{split}_mask'][:]
                
                # Filter by patient IDs if specified
                if patient_ids is not None:
                    self._filter_by_patients()
                
                print(f"Loaded {split}: {len(self.radar)} samples")
        except Exception as e:
            print(f"Error loading {split} split: {e}")
            self.radar = []

    def _filter_by_patients(self):
        """
        Filter dataset to include only specified patient IDs.
        This assumes patient information is encoded in flags.
        """
        if len(self.flags) == 0:
            return
        
        # Extract patient IDs from flags (assuming first column contains patient ID)
        if self.flags.ndim > 1:
            sample_patient_ids = self.flags[:, 0]
        else:
            sample_patient_ids = self.flags
        
        # Create boolean mask for samples belonging to specified patients
        mask = np.isin(sample_patient_ids, self.patient_ids)
        
        # Apply filter
        self.radar = self.radar[mask]
        self.ecg = self.ecg[mask]
        self.icg = self.icg[mask]
        self.dicg = self.dicg[mask]
        self.bp = self.bp[mask]
        self.strain = self.strain[mask]
        self.resp = self.resp[mask]
        self.flags = self.flags[mask]
        self.mask = self.mask[mask]
        
        print(f"Filtered to {len(self.radar)} samples for patients {self.patient_ids}")

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
            'strain': to_tensor(self.strain[idx]),
            'resp': to_tensor(self.resp[idx]),
            'flags': torch.from_numpy(self.flags[idx]).float(),
            'mask': to_tensor(self.mask[idx])
        }

def get_patient_ids_from_h5(h5_path):
    """
    Extract unique patient IDs from all splits in the H5 file.
    """
    patient_ids = set()
    
    try:
        with h5py.File(h5_path, 'r') as f:
            for split in ['train', 'val', 'test']:
                if f'{split}_flags' in f:
                    flags = f[f'{split}_flags'][:]
                    if flags.ndim > 1:
                        split_patient_ids = set(flags[:, 0])
                    else:
                        split_patient_ids = set(flags)
                    patient_ids.update(split_patient_ids)
                    print(f"Found {len(split_patient_ids)} patients in {split} split")
    except Exception as e:
        print(f"Error reading patient IDs: {e}")
        return []
    
    patient_list = sorted(list(patient_ids))
    print(f"Total unique patients: {len(patient_list)}")
    return patient_list

def create_loso_splits(h5_path, patient_ids):
    """
    Create LOSO splits where each patient is left out once as test set.
    """
    splits = []
    
    for test_patient in patient_ids:
        # Get training patients (all except test patient)
        train_patients = [p for p in patient_ids if p != test_patient]
        
        # For LOSO, we can use existing train/val structure
        # but filter by patient IDs
        splits.append({
            'train_patients': train_patients,
            'val_patients': train_patients,  # Use same patients for validation
            'test_patient': test_patient,
            'train_split': 'train',
            'val_split': 'val', 
            'test_split': 'test'
        })
    
    return splits

def create_loso_dataloaders(h5_path, split_info, config):
    """
    Create dataloaders for a specific LOSO split.
    """
    # Create datasets with patient filtering
    train_ds = H5RamDataset(h5_path, split_info['train_split'], 
                           patient_ids=split_info['train_patients'])
    val_ds = H5RamDataset(h5_path, split_info['val_split'], 
                         patient_ids=split_info['val_patients'])
    test_ds = H5RamDataset(h5_path, split_info['test_split'], 
                          patient_ids=[split_info['test_patient']])
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, 
                          shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, 
                           shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def create_patient_wise_splits(config):
    """
    Original function for standard train/val/test splits (kept for compatibility).
    """
    maneuver_name = config.maneuvers_to_load[0][1] 
    h5_path = config.h5_file_pattern.format(m=maneuver_name)
    
    print(f"Loading datasets from: {h5_path}")
    
    # 1. Train
    train_ds = H5RamDataset(h5_path, 'train')
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    # 2. Val
    val_ds = H5RamDataset(h5_path, 'val')
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # 3. Test
    test_ds = H5RamDataset(h5_path, 'test')
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
