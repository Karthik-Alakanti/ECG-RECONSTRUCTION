# =======================================================================
# postprocessing.py (Unified 5-Task + Auto-Zoom)
# =======================================================================

import argparse
from pathlib import Path
import csv
import numpy as np
import scipy.signal
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from configs.config import cfg
from models.cast_ecg import SimplifiedCASTECG_Paper

# =======================================================================
# Helper Functions
# =======================================================================

def normalize(sig):
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

def apply_processing(signal, fs, band=None, notch=False):
    if len(signal) == 0: return signal
    if notch:
        b, a = scipy.signal.iirnotch(50.0, 30.0, fs)
        signal = scipy.signal.filtfilt(b, a, signal)
    if band:
        b, a = scipy.signal.butter(4, [band[0]/(0.5*fs), band[1]/(0.5*fs)], "band")
        signal = scipy.signal.filtfilt(b, a, signal)
    
    # Baseline correction
    x = np.arange(len(signal))
    p = np.polyfit(x, signal, 5)
    signal = signal - np.polyval(p, x)
    
    return normalize(signal)

def derive_6_leads(lead_I, lead_II):
    """Derives III, aVR, aVL, aVF from I and II"""
    III = lead_II - lead_I
    aVR = -0.5 * (lead_I + lead_II)
    aVL = lead_I - 0.5 * lead_II
    aVF = lead_II - 0.5 * lead_I
    return III, aVR, aVL, aVF

# =======================================================================
# Data Loading
# =======================================================================

def load_unified_data(mat_path, target_fs):
    try:
        d = sio.loadmat(mat_path)
        
        fs_radar = d.get('fs_radar', [[2000]])[0, 0]
        up, down = int(target_fs), int(fs_radar)
        i_r = scipy.signal.resample_poly(d['radar_i'].flatten(), up, down)
        q_r = scipy.signal.resample_poly(d['radar_q'].flatten(), up, down)
        
        phase = apply_processing(np.unwrap(np.arctan2(q_r, i_r)), target_fs, [0.8, 8.0], notch=True)
        mag = apply_processing(np.sqrt(i_r**2 + q_r**2), target_fs, [0.8, 8.0], notch=True)
        radar_stack = np.stack([phase, mag], axis=0)

        ecg_band = [0.5, 40.0]
        e1 = apply_processing(scipy.signal.resample_poly(d['tfm_ecg1'].flatten(), up, down), target_fs, ecg_band)
        e2 = apply_processing(scipy.signal.resample_poly(d['tfm_ecg2'].flatten(), up, down), target_fs, ecg_band)
        ecg_stack = np.stack([e1, e2], axis=0) 

        fs_bp = d.get('fs_bp', [[200]])[0, 0]
        bp = apply_processing(scipy.signal.resample_poly(d['tfm_bp'].flatten(), int(target_fs), int(fs_bp)), target_fs, [0.05, 10.0])

        fs_icg = d.get('fs_icg', [[1000]])[0, 0]
        icg = apply_processing(scipy.signal.resample_poly(d['tfm_icg'].flatten(), int(target_fs), int(fs_icg)), target_fs, [0.5, 20.0])

        fs_z0 = d.get('fs_z0', [[100]])[0, 0]
        resp = apply_processing(scipy.signal.resample_poly(d['tfm_z0'].flatten(), int(target_fs), int(fs_z0)), target_fs, [0.05, 1.0])

        L = min(radar_stack.shape[1], ecg_stack.shape[1], len(bp), len(icg), len(resp))
        
        return {
            'radar': radar_stack[:, :L],
            'ecg': ecg_stack[:, :L],
            'bp': bp[:L],
            'icg': icg[:L],
            'resp': resp[:L]
        }

    except Exception as ex:
        print(f"Error loading {mat_path}: {ex}")
        return None

# =======================================================================
# Stitching
# =======================================================================

@torch.no_grad()
def stitch_unified_inference(model, radar_2ch, cfg, device):
    model.eval()
    L = radar_2ch.shape[1]
    W, S = cfg.window_size_samples, cfg.stride_samples
    win = scipy.signal.windows.hann(W)
    
    preds = {
        'ecg': np.zeros((2, L)), 
        'bp': np.zeros(L),
        'icg': np.zeros(L),
        'resp': np.zeros(L)
    }
    counts = np.zeros(L)

    print("Running Inference...")
    for i in tqdm(range(0, L - W, S)):
        x = torch.from_numpy(radar_2ch[:, i:i+W]).unsqueeze(0).float().to(device)
        out = model(x)
        
        preds['ecg'][:, i:i+W] += out['ecg'][0].cpu().numpy() * win
        preds['bp'][i:i+W] += out['bp'][0, 0].cpu().numpy() * win
        preds['icg'][i:i+W] += out['icg'][0, 0].cpu().numpy() * win
        preds['resp'][i:i+W] += out['resp'][0, 0].cpu().numpy() * win
        counts[i:i+W] += win

    mask = counts > 0
    for k in preds:
        if k == 'ecg': preds[k][:, mask] /= counts[mask]
        else: preds[k][mask] /= counts[mask]
            
    return preds

# =======================================================================
# Plotting (ZOOM FUNCTION ADDED)
# =======================================================================

def plot_zoom_window(data_dict, preds, derived_leads, pid, fs, start_idx, duration_sec, save_name):
    """
    Plots a specific zoomed-in window (default 10s)
    """
    end_idx = min(start_idx + int(duration_sec * fs), len(preds['bp']))
    t = np.arange(start_idx, end_idx) / fs
    t = t - t[0] # Zero the time axis for the plot
    
    # Slicing
    sl = slice(start_idx, end_idx)
    
    fig, axs = plt.subplots(9, 1, figsize=(16, 22), sharex=True)
    
    # 1. ECG Lead I
    axs[0].plot(t, data_dict['ecg'][0, sl], 'g', alpha=0.6, label='Truth')
    axs[0].plot(t, preds['ecg'][0, sl], 'r--', alpha=0.8, label='Pred')
    axs[0].set_ylabel("Lead I"); axs[0].legend(loc="upper right")
    axs[0].set_title(f"{pid} | Best 10s Reconstruction Window")

    # 2. ECG Lead II
    axs[1].plot(t, data_dict['ecg'][1, sl], 'g', alpha=0.6)
    axs[1].plot(t, preds['ecg'][1, sl], 'r--', alpha=0.8)
    axs[1].set_ylabel("Lead II")

    # 3-6. Derived
    labels = ['III', 'aVR', 'aVL', 'aVF']
    tr_I, tr_II = data_dict['ecg'][0, sl], data_dict['ecg'][1, sl]
    truth_derived = [tr_II - tr_I, -0.5*(tr_I+tr_II), tr_I - 0.5*tr_II, tr_II - 0.5*tr_I]
    
    for i, name in enumerate(labels):
        ax = axs[2+i]
        ax.plot(t, truth_derived[i], 'g', alpha=0.4)
        ax.plot(t, derived_leads[i][sl], 'b--', alpha=0.8)
        ax.set_ylabel(f"{name} (Calc)")

    # 7. BP
    axs[6].plot(t, data_dict['bp'][sl], 'g', alpha=0.6)
    axs[6].plot(t, preds['bp'][sl], 'm--', alpha=0.8)
    axs[6].set_ylabel("BP")

    # 8. ICG
    axs[7].plot(t, data_dict['icg'][sl], 'g', alpha=0.6)
    axs[7].plot(t, preds['icg'][sl], 'c--', alpha=0.8)
    axs[7].set_ylabel("ICG")

    # 9. Resp
    axs[8].plot(t, data_dict['resp'][sl], 'g', alpha=0.6)
    axs[8].plot(t, preds['resp'][sl], 'k--', alpha=0.8)
    axs[8].set_ylabel("Resp")
    axs[8].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.close()
    print(f"Saved Zoomed Plot: {save_name}")

def plot_full_overview(preds, pid, fs, save_name):
    """Saves the dense full-duration plot (just for record)"""
    t = np.arange(len(preds['bp'])) / fs
    plt.figure(figsize=(12, 6))
    plt.plot(t, preds['ecg'][1], 'r', lw=0.5)
    plt.title(f"{pid} Full Duration Overview (Lead II)")
    plt.xlabel("Time (s)")
    plt.savefig(save_name, dpi=150)
    plt.close()

# =======================================================================
# MAIN
# =======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True, help="Patient ID (e.g., GDN0001)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_unified.pth")
    args = parser.parse_args()

    pid = args.id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Unified Model from {args.ckpt}...")
    cfg.in_channels = 2
    model = SimplifiedCASTECG_Paper(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    mat_path = Path("../../../data") / pid / f"{pid}_1_Resting.mat"
    print(f"Loading data from {mat_path}...")
    data_dict = load_unified_data(mat_path, cfg.fs)
    if data_dict is None: return

    # Inference
    preds = stitch_unified_inference(model, data_dict['radar'], cfg, device)

    # Derive Leads
    III, aVR, aVL, aVF = derive_6_leads(preds['ecg'][0], preds['ecg'][1])
    derived_leads = [III, aVR, aVL, aVF]

    # --- FIND BEST 10s WINDOW ---
    print("Finding best 10s window for plotting...")
    L = len(preds['bp'])
    W_10s = int(10 * cfg.fs)
    step = int(1 * cfg.fs)
    
    best_score = -1
    best_idx = 0
    
    # We use Lead II PCC as the quality metric for choosing the window
    target = data_dict['ecg'][1]
    recon = preds['ecg'][1]
    
    for i in range(0, L - W_10s, step):
        t_win = target[i:i+W_10s]
        r_win = recon[i:i+W_10s]
        score = np.corrcoef(t_win, r_win)[0, 1]
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    print(f"Best Window found at {best_idx/cfg.fs:.2f}s (PCC={best_score:.4f})")

    # --- PLOTTING ---
    # 1. Full Overview (The "Messy" one, but useful for macro view)
    plot_full_overview(preds, pid, cfg.fs, f"{pid}_full_overview.png")

    # 2. Zoomed 10s (The "Clean" one)
    plot_zoom_window(
        data_dict, preds, derived_leads, 
        pid, cfg.fs, 
        start_idx=best_idx, 
        duration_sec=10, 
        save_name=f"{pid}_zoom_10s_best.png"
    )

if __name__ == "__main__":
    main()