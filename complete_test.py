"""
Standard Test Script (7-Task Unified Model)
Evaluates on the unseen Test Split (GDN0027-GDN0030).
"""
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# Project Imports
from configs.config import cfg
from models.cast_ecg import SimplifiedCASTECG_Paper as SimplifiedCASTECG
from dataload.dataset import create_patient_wise_splits
from utils.metrics import calculate_all_metrics

def save_test_plot(preds, truths, sample_idx, save_dir):
    """
    Generates the 11-Row Clinical Plot:
    Row 1-6: ECG Leads (I, II, III, aVR, aVL, aVF)
    Row 7: Blood Pressure
    Row 8: ICG (Impedance)
    Row 9: Flow Velocity (dICG/dt)
    """
    fs = cfg.fs
    t = np.arange(preds['ecg'].shape[-1]) / fs
    
    # 6 Rows for 5 signals (2 ECG + BP + ICG + Flow)
    fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
    
    # --- 1. ECG (2 Leads) ---
    leads = ['I', 'II']
    for j, name in enumerate(leads):
        # Truth (Green), Pred (Red Dashed)
        axes[j].plot(t, truths['ecg'][j], 'g', alpha=0.6, label='Truth' if j==0 else "")
        axes[j].plot(t, preds['ecg'][j], 'r--', alpha=0.8, label='Pred' if j==0 else "")
        axes[j].set_ylabel(name)
        axes[j].grid(True, alpha=0.3)
        if j == 0: axes[j].legend(loc="upper right")

    # --- 2. Hemodynamics & Vitals ---
    # BP (Row 2)
    axes[2].plot(t, truths['bp'][0], 'g', alpha=0.6)
    axes[2].plot(t, preds['bp'][0], 'm--', alpha=0.8)
    axes[2].set_ylabel("BP (mmHg)")
    
    # ICG (Row 3)
    axes[3].plot(t, truths['icg'][0], 'g', alpha=0.6)
    axes[3].plot(t, preds['icg'][0], 'c--', alpha=0.8)
    axes[3].set_ylabel("ICG (Vol)")
    
    # Flow (Row 4)
    axes[4].plot(t, truths['dicg'][0], 'g', alpha=0.6)
    axes[4].plot(t, preds['dicg'][0], 'k--', alpha=0.8)
    axes[4].set_ylabel("Flow (Vel)")
    
    axes[5].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"test_sample_{sample_idx}.png"))
    plt.close()

def main():
    # 1. Setup
    device = torch.device(cfg.device)
    test_save_dir = os.path.join(cfg.results_dir, 'standard_test_evaluation')
    os.makedirs(test_save_dir, exist_ok=True)
    
    print("="*60)
    print(f"Running STANDARD TEST on {device}")
    print("Evaluating 7-Task Performance on Unseen Patients")
    print("="*60)

    # 2. Load Data (Test Split Only)
    print("Loading Test Data...")
    _, _, test_loader = create_patient_wise_splits(cfg)
    
    if len(test_loader) == 0:
        print("Error: Test loader is empty. Check your dataset splitting.")
        return

    # 3. Load Model
    model = SimplifiedCASTECG(cfg).to(device)
    
    # Try loading best final, fallback to best ecg
    ckpt_path = f"{cfg.checkpoint_dir}/best_final.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"{cfg.checkpoint_dir}/best_ecg_model.pth"
    
    if not os.path.exists(ckpt_path):
        print(f"Error: No checkpoint found at {cfg.checkpoint_dir}")
        return
        
    print(f"Loading weights from: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 4. Inference & Metrics Loop
    # Initialize log dictionary
    logs = {
        'ECG_PCC': [], 'ECG_RMSE': [],
        'BP_SBP_MAE': [], 'BP_DBP_MAE': [], 'BP_PCC': [],
        'ICG_PCC': [], 'Flow_PCC': [],
        'Strain_MAE': [],
        'Resp_RPM_Error': [], 'Resp_PCC': []
    }
    
    print("Starting Inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            # --- INPUT: STRICTLY RADAR ONLY ---
            radar_input = batch['radar_i'].to(device)
            
            # Forward Pass
            outputs = model(radar_input)
            
            # --- EVALUATION ---
            # Get Targets (numpy) for metrics
            targets = {k: batch[k].cpu().numpy() for k in ['ecg', 'bp', 'icg', 'dicg']}
            
            # Get Preds (numpy)
            preds = {k: outputs[k].cpu().numpy() for k in ['ecg', 'bp', 'icg', 'dicg']}
            
            # Map 'ecg' to 'ecg_6lead' for the metrics calculator (since our output is already 6 leads)
            preds['ecg_6lead'] = preds['ecg']
            targets['ecg_6lead'] = targets['ecg']

            # Iterate through batch
            B = radar_input.shape[0]
            for i in range(B):
                # Slice single sample for metrics
                p_sample = {k: v[i] for k,v in preds.items()}
                t_sample = {k: v[i] for k,v in targets.items()}
                
                # Compute Metrics (using utils.metrics)
                m = calculate_all_metrics(p_sample, t_sample, cfg.fs)
                
                # Log results
                for k, v in m.items():
                    logs[k].append(v)
                
                # Plot first 20 samples of the test set
                global_idx = batch_idx * cfg.batch_size + i
                if global_idx < 20:
                    # Remove 'ecg_6lead' key for plotting function compatibility if needed, 
                    # but save_test_plot uses 'ecg' key which is fine.
                    save_test_plot(p_sample, t_sample, global_idx, test_save_dir)

    # 5. Final Report
    print("\n" + "="*50)
    print("FINAL TEST RESULTS (Mean ± Std)")
    print("="*50)
    
    # Calculate Mean & Std for all logs
    results = {k: (np.mean(v), np.std(v)) for k,v in logs.items()}
    
    print(f"1. ECG (2-Leads: I, II):")
    print(f"   Shape (PCC):          {results['ECG_PCC'][0]:.4f} ± {results['ECG_PCC'][1]:.4f}")
    print(f"   Error (RMSE):         {results['ECG_RMSE'][0]:.4f}")
    
    print(f"\n2. Blood Pressure:")
    print(f"   Systolic Error:       {results['BP_SBP_MAE'][0]:.2f} ± {results['BP_SBP_MAE'][1]:.2f} mmHg")
    print(f"   Diastolic Error:      {results['BP_DBP_MAE'][0]:.2f} ± {results['BP_DBP_MAE'][1]:.2f} mmHg")
    print(f"   Shape (PCC):          {results['BP_PCC'][0]:.4f}")

    print(f"\n3. Hemodynamics:")
    print(f"   ICG Shape (PCC):      {results['ICG_PCC'][0]:.4f}")
    print(f"   Flow Velocity (PCC):  {results['Flow_PCC'][0]:.4f}")
    print("="*50)
    
    # Save Raw Metrics to CSV
    df = pd.DataFrame(logs)
    csv_path = os.path.join(test_save_dir, "test_metrics_detailed.csv")
    df.to_csv(csv_path, index=False)
    print(f"Detailed metrics saved to: {csv_path}")
    print(f"Plots saved to: {test_save_dir}")

if __name__ == "__main__":
    main()