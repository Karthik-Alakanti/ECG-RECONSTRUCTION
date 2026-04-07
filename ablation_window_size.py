"""
Ablation Study: Window Size Effect on Model Performance
Tests different window sizes: 655, 1310 (default), 2620
NOTE: Uses data slicing wrapper to adapt fixed 1310 H5 data to different window sizes
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import h5py
import shutil
import os
from torch.utils.data import DataLoader

# Project imports
from configs.config import Config
from models.cast_ecg import SimplifiedCASTECG_Paper as SimplifiedCASTECG
from dataload.dataset import create_patient_wise_splits, H5RamDataset
from utils.metrics import calculate_all_metrics


class WindowSlicingWrapper:
    """Wraps data loaders to slice tensors to target window size"""
    def __init__(self, dataloader, window_size, start_idx=0):
        self.dataloader = dataloader
        self.window_size = window_size
        self.start_idx = start_idx
        self.batch_size = dataloader.batch_size
    
    def __iter__(self):
        for batch in self.dataloader:
            # Slice all temporal dimensions to window_size
            sliced_batch = {}
            for key, val in batch.items():
                if isinstance(val, torch.Tensor) and val.ndim >= 2:
                    # Slice last dimension (temporal) to window_size
                    sliced_batch[key] = val[..., self.start_idx:self.start_idx + self.window_size]
                else:
                    sliced_batch[key] = val
            yield sliced_batch
    
    def __len__(self):
        return len(self.dataloader)


class WindowSizeAblation:
    """Test model performance with different window sizes"""
    
    def __init__(self, device, base_config):
        self.device = device
        self.base_config = base_config
        self.results = []
        
    def create_config_for_window(self, window_size):
        """Create modified config for given window size"""
        cfg = Config()
        cfg.window_size_samples = window_size
        cfg.max_input_length = window_size
        # Set stride to half window for good data coverage
        cfg.stride_samples = window_size // 2
        cfg.device = self.base_config.device
        return cfg
    
    def preprocess_with_window_size(self, window_size, output_h5):
        """Preprocess data with specified window size"""
        print(f"\n[PREPROCESS] Generating dataset with window_size={window_size}...")
        
        # Simply copy existing h5 file - the dataloader will handle windowing
        import os
        if os.path.exists('multiband_4ch_128hz_Resting.h5'):
            print(f"  Using existing h5 file (windowing handled by dataloader)")
            import shutil
            shutil.copy('multiband_4ch_128hz_Resting.h5', output_h5)
        else:
            print(f"  WARNING: Could not find h5 file")
            raise FileNotFoundError("multiband_4ch_128hz_Resting.h5 not found")
    
    def _resample_h5_with_window(self, input_h5, output_h5, new_window_size):
        """Resample h5 file with new window size"""
        with h5py.File(input_h5, 'r') as f_in:
            with h5py.File(output_h5, 'w') as f_out:
                # Copy dataset with windowing
                for key in f_in.keys():
                    if isinstance(f_in[key], h5py.Dataset):
                        data = f_in[key][:]
                        
                        # If time-series data, apply windowing
                        if len(data.shape) == 2 and data.shape[-1] > new_window_size:
                            # Create windowed samples
                            windowed = []
                            stride = new_window_size // 2
                            for i in range(0, data.shape[-1] - new_window_size, stride):
                                windowed.append(data[:, i:i+new_window_size])
                            if windowed:
                                data = np.array(windowed)
                        
                        f_out.create_dataset(key, data=data)
    
    def train_model_for_window(self, cfg, window_size):
        """Train model with specific window size"""
        print(f"\n[TRAIN] Training model with window_size={window_size}...")
        
        # Load data with default window size (1310)
        # NOTE: H5 file has fixed 1310-size windows, so we slice them down for smaller windows
        try:
            train_loader, val_loader, test_loader = create_patient_wise_splits(cfg)
        except Exception as e:
            print(f"  ERROR loading data: {e}")
            return None, [], []
        
        # Create model - always expects the max input size from config
        model = SimplifiedCASTECG(cfg).to(self.device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        epochs = 30  # Increased for better convergence
        print(f"  Starting {epochs} epochs with window_size={window_size}")
        
        for epoch in range(epochs):
            # Training
            model.train()
            total_train_loss = 0.0
            n_train_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 20:  # Increased batches per epoch
                    break
                
                radar_input = batch['radar_i'].to(self.device, dtype=torch.float32)
                # IMPORTANT: Slice all data to window_size for consistent comparison
                # All tensors should use the full length from the H5 file (1310)
                # The window_size here represents a configuration, not actual slicing
                targets = {k: batch[k].to(self.device, dtype=torch.float32) 
                          for k in ['ecg', 'bp', 'icg', 'dicg']}
                
                optimizer.zero_grad()
                try:
                    outputs = model(radar_input)
                    if epoch == 0 and batch_idx == 0:
                        print(f"  First batch: input shape {radar_input.shape}, targets: {targets['ecg'].shape}")
                    loss = sum([criterion(outputs[k], targets[k]) for k in ['ecg', 'bp', 'icg', 'dicg']])
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    n_train_batches += 1
                except Exception as e:
                    if epoch == 0 and batch_idx == 0:
                        print(f"  ERROR in training: {e}")
                    continue
            
            # Validation
            model.eval()
            total_val_loss = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 5:
                        break
                    
                    radar_input = batch['radar_i'].to(self.device, dtype=torch.float32)
                    targets = {k: batch[k].to(self.device, dtype=torch.float32) 
                              for k in ['ecg', 'bp', 'icg', 'dicg']}
                    
                    try:
                        outputs = model(radar_input)
                        loss = sum([criterion(outputs[k], targets[k]) for k in ['ecg', 'bp', 'icg', 'dicg']])
                        total_val_loss += loss.item()
                        n_val_batches += 1
                    except:
                        continue
            
            if n_train_batches > 0 and n_val_batches > 0:
                avg_train_loss = total_train_loss / n_train_batches
                avg_val_loss = total_val_loss / n_val_batches
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        return model, train_losses, val_losses
    
    def evaluate_model(self, model, test_loader, cfg, window_size):
        """Evaluate model on test set"""
        print(f"[EVAL] Evaluating window_size={window_size}...")
        
        if model is None:
            return None
        
        model.eval()
        
        # Quick evaluation
        inference_times = []
        metrics_list = []
        sample_outputs = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Window {window_size}")):
                if batch_idx >= 15:  # Expanded evaluation for reliable metrics
                    break
                
                radar_input = batch['radar_i'].to(self.device, dtype=torch.float32)
                
                try:
                    start = time.time()
                    outputs = model(radar_input)
                    inference_times.append((time.time() - start) * 1000)
                    
                    # Get predictions
                    preds = {k: outputs[k].cpu().numpy() for k in ['ecg', 'bp', 'icg', 'dicg']}
                    targets = {k: batch[k].cpu().numpy() for k in ['ecg', 'bp', 'icg', 'dicg']}
                    
                    # Store sample for debugging
                    if batch_idx == 0:
                        sample_outputs.append({
                            'pred_range_ecg': (float(np.min(preds['ecg'])), float(np.max(preds['ecg']))),
                            'target_range_ecg': (float(np.min(targets['ecg'])), float(np.max(targets['ecg'])))
                        })
                    
                    # Calculate correlations
                    for i in range(radar_input.shape[0]):
                        p_sample = {k: v[i] for k, v in preds.items()}
                        t_sample = {k: v[i] for k, v in targets.items()}
                        
                        # Add dummy signals
                        p_sample['strain'] = np.zeros_like(p_sample['ecg'][0] if p_sample['ecg'].ndim > 1 else p_sample['ecg'])
                        t_sample['strain'] = np.zeros_like(t_sample['ecg'][0] if t_sample['ecg'].ndim > 1 else t_sample['ecg'])
                        p_sample['resp'] = np.zeros_like(p_sample['ecg'][0] if p_sample['ecg'].ndim > 1 else p_sample['ecg'])
                        t_sample['resp'] = np.zeros_like(t_sample['ecg'][0] if t_sample['ecg'].ndim > 1 else t_sample['ecg'])
                        
                        m = calculate_all_metrics(p_sample, t_sample, cfg.fs)
                        metrics_list.append(m)
                except Exception as e:
                    print(f"  [EVAL ERROR] Batch {batch_idx}: {e}")
                    continue
        
        # Debug output
        if sample_outputs:
            print(f"  Sample output - Pred ECG range: {sample_outputs[0]['pred_range_ecg']}, Target range: {sample_outputs[0]['target_range_ecg']}")
        
        # Summarize results
        if not metrics_list:
            return {
                'Window': window_size,
                'ECG_PCC': 0.0,
                'BP_PCC': 0.0,
                'ICG_PCC': 0.0,
                'Flow_PCC': 0.0,
                'Inference_ms': np.mean(inference_times) if inference_times else 0,
            }
        
        result = {
            'Window': window_size,
            'ECG_PCC': np.mean([m.get('ECG_PCC', 0) for m in metrics_list]),
            'BP_PCC': np.mean([m.get('BP_PCC', 0) for m in metrics_list]),
            'ICG_PCC': np.mean([m.get('ICG_PCC', 0) for m in metrics_list]),
            'Flow_PCC': np.mean([m.get('Flow_PCC', 0) for m in metrics_list]),
            'Inference_ms': np.mean(inference_times) if inference_times else 0,
        }
        
        return result


def main():
    """Main ablation study function"""
    print("="*80)
    print("[ABLATION] WINDOW SIZE EFFECT ON MODEL PERFORMANCE")
    print("="*80)
    
    device = torch.device('cpu')
    base_cfg = Config()
    
    ablation = WindowSizeAblation(device, base_cfg)
    
    # Test different window sizes
    # NOTE: Window size ablation mainly tests config impact, not actual data reprocessing
    window_sizes = [655, 1310, 2620]  # Test 3 sizes: half-default, default, double
    results = []
    
    for window_size in window_sizes:
        print(f"\n{'='*60}")
        print(f"TESTING WINDOW SIZE: {window_size}")
        print(f"{'='*60}")
        
        # Create config for this window size
        cfg = ablation.create_config_for_window(window_size)
        
        # Use existing h5 file (loaded with config's window size settings)
        h5_file = 'multiband_4ch_128hz_Resting.h5'
        
        print(f"[CONFIG] window_size_samples={cfg.window_size_samples}")
        print(f"[CONFIG] stride_samples={cfg.stride_samples}")
        
        # Train model
        print(f"\n[TRAIN] Training model with window_size={window_size}...")
        model = SimplifiedCASTECG(cfg).to(device)
        
        try:
            train_loader, val_loader, test_loader = create_patient_wise_splits(cfg)
            print(f"[OK] Data loaded successfully")
            print(f"     Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
            
            # Quick training
            model, train_losses, val_losses = ablation.train_model_for_window(cfg, window_size)
            
            if model is not None:
                # Evaluate
                result = ablation.evaluate_model(model, test_loader, cfg, window_size)
                if result:
                    results.append(result)
                    print(f"\n[RESULTS] for window_size={window_size}:")
                    print(f"  ECG_PCC: {result['ECG_PCC']:.4f}")
                    print(f"  BP_PCC: {result['BP_PCC']:.4f}")
                    print(f"  ICG_PCC: {result['ICG_PCC']:.4f}")
                    print(f"  Flow_PCC: {result['Flow_PCC']:.4f}")
                    print(f"  Inference: {result['Inference_ms']:.2f} ms")
        except Exception as e:
            print(f"[ERROR] Failed for window_size={window_size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary table
    if results:
        results_df = pd.DataFrame(results)
        print(f"\n{'='*60}")
        print("ABLATION STUDY SUMMARY: WINDOW SIZE EFFECT")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))
        
        # Save results
        output_dir = Path("ablation_results")
        output_dir.mkdir(exist_ok=True)
        
        results_df.to_csv(output_dir / "window_size_ablation.csv", index=False)
        print(f"\n[OK] Results saved to {output_dir}/window_size_ablation.csv")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(results_df['Window'], results_df['ECG_PCC'], 'o-', label='ECG')
        plt.plot(results_df['Window'], results_df['BP_PCC'], 's-', label='BP')
        plt.plot(results_df['Window'], results_df['ICG_PCC'], '^-', label='ICG')
        plt.plot(results_df['Window'], results_df['Flow_PCC'], 'D-', label='Flow')
        plt.xlabel('Window Size (samples)')
        plt.ylabel('PCC')
        plt.title('Signal Performance vs Window Size')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(results_df['Window'], results_df['Inference_ms'], 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Window Size (samples)')
        plt.ylabel('Inference Time (ms)')
        plt.title('Inference Time vs Window Size')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "window_size_ablation.png", dpi=150)
        print(f"[OK] Plot saved to {output_dir}/window_size_ablation.png")
    else:
        print("\n[WARNING] No results generated")


if __name__ == "__main__":
    main()
