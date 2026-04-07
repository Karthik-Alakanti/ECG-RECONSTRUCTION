"""
Visualization Tools for CAST-ECG
- Plot reconstruction examples
- Plot router gate evolution
- Analyze branch specialization
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from configs.config import cfg
from models.cast_ecg import SimplifiedCASTECG
from dataload.dataset import create_patient_wise_splits

@torch.no_grad()
def plot_reconstruction(model, loader, device, save_dir, n_samples=5):
    """
    Plot and save N reconstruction examples
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    batch = next(iter(loader))
    radar_i = batch['radar_i'].to(device)
    radar_q = batch['radar_q'].to(device)
    ecg_true = batch['ecg'].to(device)
    
    # Get model predictions
    outputs = model(radar_i, radar_q, epoch=cfg.epochs)
    ecg_pred = outputs['ecg']
    gates_list = outputs['gates']
    
    # Move to CPU for plotting
    ecg_true_np = ecg_true.cpu().numpy()
    ecg_pred_np = ecg_pred.cpu().numpy()
    
    time_axis = np.arange(cfg.chunk_length) / cfg.fs
    
    print(f"Saving {n_samples} reconstruction plots to {save_dir}...")
    
    for i in range(n_samples):
        true_i = ecg_true_np[i, 0]
        pred_i = ecg_pred_np[i, 0]
        
        # Calculate metrics for title
        l1 = np.mean(np.abs(pred_i - true_i))
        pearson = np.corrcoef(pred_i, true_i)[0, 1]
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        fig.suptitle(f"Sample {i} | L1: {l1:.4f} | Pearson: {pearson:.4f}", fontsize=16)
        
        # 1. Reconstruction
        axes[0].plot(time_axis, true_i, label='Ground Truth', linewidth=2)
        axes[0].plot(time_axis, pred_i, label='Prediction', linewidth=2, alpha=0.8)
        axes[0].set_title('ECG Reconstruction')
        axes[0].set_xlabel('Time (s)')
        axes[0].legend()
        axes[0].grid(True)
        
        # 2. Error
        error = pred_i - true_i
        axes[1].plot(time_axis, error, color='red')
        axes[1].set_title('Reconstruction Error')
        axes[1].set_xlabel('Time (s)')
        axes[1].grid(True)
        
        # 3. Router gates (finest scale)
        gates_finest = gates_list[0][i].cpu().numpy() # [T, 2]
        gate_time = np.linspace(0, time_axis[-1], gates_finest.shape[0])
        axes[2].plot(gate_time, gates_finest[0, :], label='Wavelet-Packet Weight', linewidth=2)
        axes[2].plot(gate_time, gates_finest[1, :], label='Filterbank Weight', linewidth=2)
        axes[2].set_title('Router Gate Weights (Scale 1)')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_ylim([0, 1])
        
        # 4. Router gates (coarsest scale)
        gates_coarsest = gates_list[-1][i].cpu().numpy() # [T, 2]
        gate_time = np.linspace(0, time_axis[-1], gates_coarsest.shape[0])
        axes[3].plot(gate_time, gates_coarsest[0, :], label='Wavelet-Packet Weight', linewidth=2)
        axes[3].plot(gate_time, gates_coarsest[1, :], label='Filterbank Weight', linewidth=2)
        axes[3].set_title('Router Gate Weights (Scale 4 - Coarsest)')
        axes[3].set_xlabel('Time (s)')
        axes[3].legend()
        axes[3].grid(True)
        axes[3].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"reconstruction_sample_{i}.png"), dpi=300)
        plt.close(fig)

def main():
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    try:
        _, val_loader, _ = create_patient_wise_splits(cfg)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update 'data_root' in configs/config.py")
        return

    # --- Model ---
    model = SimplifiedCASTECG(cfg).to(device)
    
    # Load best checkpoint
    model_path = cfg.best_model_path
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please run train.py first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded best model from {model_path}")

    # --- Plotting ---
    plot_reconstruction(
        model, 
        val_loader, 
        device, 
        save_dir=cfg.figures_dir, 
        n_samples=5
    )

if __name__ == "__main__":
    main()