"""
Training Script (Detailed Metrics & ECG-Focused Saving)
"""
import torch
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from configs.config import cfg
from models.cast_ecg import SimplifiedCASTECG_Paper as SimplifiedCASTECG
from dataload.dataset import create_patient_wise_splits
from utils.losses import CompleteLoss

def save_plots(preds, truths, epoch, save_root):
    epoch_dir = Path(save_root) / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    fs = cfg.fs
    
    for i in range(min(4, len(preds['ecg']))):
        fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
        t = np.arange(preds['ecg'].shape[-1]) / fs
        
        # Leads (only I and II now)
        leads = ['I', 'II']
        for j, name in enumerate(leads):
            axes[j].plot(t, truths['ecg'][i, j].flatten(), 'g', alpha=0.6)
            axes[j].plot(t, preds['ecg'][i, j].flatten(), 'r--', alpha=0.8)
            axes[j].set_ylabel(name)
        
        # Hemo (removed strain and resp rows)
        axes[2].plot(t, truths['bp'][i, 0].flatten(), 'g'); axes[2].plot(t, preds['bp'][i, 0].flatten(), 'm--'); axes[2].set_ylabel("BP")
        axes[3].plot(t, truths['icg'][i, 0].flatten(), 'g'); axes[3].plot(t, preds['icg'][i, 0].flatten(), 'c--'); axes[3].set_ylabel("ICG")
        axes[4].plot(t, truths['dicg'][i, 0].flatten(), 'g'); axes[4].plot(t, preds['dicg'][i, 0].flatten(), 'k--'); axes[4].set_ylabel("Flow")
        axes[5].set_xlabel("Time (s)")
        
        plt.tight_layout()
        plt.savefig(epoch_dir / f"sample_{i}.png")
        plt.close()

def train_one_epoch(model, loader, optimizer, loss_fn, epoch, device, scaler):
    model.train()
    metrics_accum = {}
    total_loss = 0.0
    
    for batch in tqdm(loader, desc=f"Epoch {epoch} [Train]"):
        radar = batch['radar_i'].to(device)
        mask = batch['mask'].to(device)
        flags = batch['flags'].to(device)
        targets = {k: batch[k].to(device) for k in ['ecg', 'icg', 'dicg', 'bp']}
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(radar)
            loss_dict = loss_fn(outputs, targets, mask, flags)
        
        scaler.scale(loss_dict['total']).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss_dict['total'].item()
        
        # Accumulate metrics
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor): v = v.item()
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v

    # Average metrics
    avg_metrics = {k: v / len(loader) for k, v in metrics_accum.items()}
    return total_loss / len(loader), avg_metrics

@torch.no_grad()
def validate(model, loader, loss_fn, epoch, device):
    model.eval()
    metrics_accum = {}
    plot_preds, plot_truths = {}, {}
    
    for i, batch in enumerate(tqdm(loader, desc="[Val]")):
        radar = batch['radar_i'].to(device)
        mask = batch['mask'].to(device)
        flags = batch['flags'].to(device)
        targets = {k: batch[k].to(device) for k in ['ecg', 'icg', 'dicg', 'bp']}
        
        outputs = model(radar)
        loss_dict = loss_fn(outputs, targets, mask, flags)
        
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor): v = v.item()
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
            
        if i == 0:
            for k in targets:
                plot_preds[k] = outputs[k].cpu().numpy()
                plot_truths[k] = targets[k].cpu().numpy()
            plot_preds['ecg_6lead'] = outputs['ecg'].cpu().numpy()

    avg_metrics = {k: v / len(loader) for k, v in metrics_accum.items()}
    return avg_metrics, plot_preds, plot_truths

def main():
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)
    device = torch.device(cfg.device)
    
    train_loader, val_loader, _ = create_patient_wise_splits(cfg)
    model = SimplifiedCASTECG(cfg).to(device)
    loss_fn = CompleteLoss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    print("Starting Training (Saving Best Temporal+Spectral ECG)...")
    best_ecg_score = -1.0

    for epoch in range(1, cfg.epochs + 1):
        t_loss, t_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, epoch, device, scaler)
        v_metrics, p_preds, p_truths = validate(model, val_loader, loss_fn, epoch, device)
        
        scheduler.step()
        
        # --- 1. PRINT DETAILED METRICS ---
        print(f"\n{'='*20} EPOCH {epoch} SUMMARY {'='*20}")
        print(f"{'METRIC':<20} | {'TRAIN':<10} | {'VAL':<10}")
        print("-" * 46)
        
        keys = ['ecg_temp_score', 'ecg_spec_score', 'ecg_l1_score', 
                'bp_corr_score', 'icg_corr_score', 'flow_corr_score']
        
        for k in keys:
            t_val = t_metrics.get(k, 0.0)
            v_val = v_metrics.get(k, 0.0)
            print(f"{k:<20} | {t_val:>9.2f}  | {v_val:>9.2f}")
            
        print("-" * 46)
        print(f"Total Loss           | {t_loss:>9.4f}  | {v_metrics['total']:>9.4f}")
        print("=" * 46)

        # --- 2. SAVE PLOTS ---
        save_plots(p_preds, p_truths, epoch, cfg.figures_dir)
        
        # --- 3. SAVE BEST MODEL (Temporal + Spectral) ---
        # We average Temporal and Spectral scores for ECG
        current_ecg_combined = (v_metrics['ecg_temp_score'] + v_metrics['ecg_spec_score']) / 2.0
        
        if current_ecg_combined > best_ecg_score:
            best_ecg_score = current_ecg_combined
            torch.save(model.state_dict(), f"{cfg.checkpoint_dir}/best_ecg_model.pth")
            print(f"  ⭐ NEW BEST MODEL! (ECG Combined Score: {current_ecg_combined:.2f})")
            
        # Optional: Save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{cfg.checkpoint_dir}/epoch_{epoch}.pth")

if __name__ == "__main__":
    main()