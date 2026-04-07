# =======================================================================
# postprocess.py (NaN-Proof Debug Version)
# Usage: python postprocess.py --id GDN0030 --ckpt checkpoints_multiband/best_ecg_model.pth
# =======================================================================

import argparse
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal
from pathlib import Path
import os
import sys

from configs.config import cfg
from models.cast_ecg import SimplifiedCASTECG_Paper as SimplifiedCASTECG

# --- 0. SAFETY HELPERS ---
def check_model_health(model):
    print("Checking model weights...")
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"CRITICAL ERROR: Layer '{name}' contains NaNs!")
            has_nan = True
    if has_nan:
        print("THE CHECKPOINT IS CORRUPTED (Exploding Gradients). You must retrain.")
        sys.exit(1)
    print("Model weights look healthy.")

def safe_normalize(signal):
    # Replace NaNs with 0
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(signal)
    if std < 1e-6: return np.zeros_like(signal)
    return (signal - np.mean(signal)) / (std + 1e-8)

def apply_processing(signal, fs, band=None, notch=False, baseline=True):
    if len(signal) == 0: return signal
    
    # 1. Sanitize Input
    signal = np.nan_to_num(signal)
    
    if notch:
        b, a = scipy.signal.iirnotch(50.0, 30.0, fs)
        signal = scipy.signal.filtfilt(b, a, signal)
    if band:
        b, a = scipy.signal.butter(4, [band[0]/(0.5*fs), band[1]/(0.5*fs)], "band")
        signal = scipy.signal.filtfilt(b, a, signal)
    if baseline:
        x = np.arange(len(signal))
        # Catch Polyfit errors on flat lines
        try:
            p = np.polyfit(x, signal, 5)
            signal = signal - np.polyval(p, x)
        except:
            signal = signal - np.mean(signal)
            
    return safe_normalize(signal)

def process_patient_data_on_the_fly(mat_path):
    d = sio.loadmat(mat_path)
    target_fs = cfg.fs
    
    fs_radar = d.get('fs_radar', [[2000]])[0, 0]
    up, down = int(target_fs), int(fs_radar)
    
    # Raw Load
    i_r = d['radar_i'].flatten()
    q_r = d['radar_q'].flatten()
    
    # Sanitize RAW data
    i_r = np.nan_to_num(i_r)
    q_r = np.nan_to_num(q_r)
    
    i_r = scipy.signal.resample_poly(i_r, up, down)
    q_r = scipy.signal.resample_poly(q_r, up, down)
    
    raw_phase = np.unwrap(np.arctan2(q_r, i_r))
    raw_mag = np.sqrt(i_r**2 + q_r**2)
    
    # 1. Heart Band (0.8 - 8.0 Hz)
    p_heart = apply_processing(raw_phase, target_fs, [0.8, 8.0], notch=True, baseline=True)
    m_heart = apply_processing(raw_mag, target_fs, [0.8, 8.0], notch=True, baseline=True)
    
    # 2. Resp Band (0.05 - 0.8 Hz)
    p_resp = apply_processing(raw_phase, target_fs, [0.05, 0.8], notch=False, baseline=False)
    m_resp = apply_processing(raw_mag, target_fs, [0.05, 0.8], notch=False, baseline=False)
    
    radar = np.stack([p_heart, m_heart, p_resp, m_resp])
    
    if np.isnan(radar).any():
        print("WARNING: NaNs detected in processed radar input! Zeroing out.")
        radar = np.nan_to_num(radar)
        
    return radar.astype(np.float32)

# --- Clinical Helpers ---
def normalize_minmax(x):
    x = np.nan_to_num(x)
    ma, mi = np.max(x), np.min(x)
    if (ma - mi) < 1e-6: return np.zeros_like(x)
    return (x - mi) / (ma - mi + 1e-8)

def detect_peaks_robust(signal, fs):
    sig_norm = normalize_minmax(signal)
    b, a = scipy.signal.butter(2, [8/(0.5*fs), 20/(0.5*fs)], "band")
    filt = scipy.signal.filtfilt(b, a, sig_norm)
    integ = np.convolve(np.gradient(filt)**2, np.ones(int(0.1*fs))/int(0.1*fs), 'same')
    thresh = np.max(integ) * 0.3 
    peaks, _ = scipy.signal.find_peaks(integ, height=thresh, distance=int(0.4*fs))
    return peaks

def calculate_hemodynamics(flow_sig, r_peaks, fs):
    metrics = {'SV_est': [], 'Contractility': [], 'LVET': []}
    if len(r_peaks) < 2: return {k: 0.0 for k in metrics}
    flow_sig = normalize_minmax(flow_sig)
    for r in r_peaks:
        win_start, win_end = r, min(len(flow_sig), r + int(0.35*fs))
        if win_end - win_start < 10: continue
        window = flow_sig[win_start:win_end]
        c_amp = np.max(window)
        half_max = c_amp / 2
        above = np.where(window > half_max)[0]
        lvet_ms = ((above[-1]-above[0])/fs)*1000 if len(above)>0 else 0
        metrics['Contractility'].append(c_amp)
        metrics['LVET'].append(lvet_ms)
        metrics['SV_est'].append(c_amp * lvet_ms)
    return {k: np.mean(v) if len(v)>0 else 0.0 for k, v in metrics.items()}

def plot_colored_ecg(pid, ecg, flow, r_peaks, fs, save_dir):
    t = np.arange(len(ecg)) / fs
    ecg = normalize_minmax(ecg) 
    
    p_mask = np.zeros_like(ecg, dtype=bool)
    qrs_mask = np.zeros_like(ecg, dtype=bool)
    t_mask = np.zeros_like(ecg, dtype=bool)
    
    p_s, p_e = int(0.2*fs), int(0.1*fs)
    q_s, q_e = int(0.05*fs), int(0.05*fs)
    t_s, t_e = int(0.1*fs), int(0.4*fs)
    
    for r in r_peaks:
        p_mask[max(0, r-p_s):max(0, r-p_e)] = True
        qrs_mask[max(0, r-q_s):min(len(ecg), r+q_e)] = True
        t_mask[min(len(ecg), r+t_s):min(len(ecg), r+t_e)] = True

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    ax1.plot(t, ecg, color='lightgray', linewidth=1, alpha=0.5)
    e_p = ecg.copy(); e_p[~p_mask] = np.nan
    ax1.plot(t, e_p, color='blue', linewidth=2, label='P-Wave')
    e_q = ecg.copy(); e_q[~qrs_mask] = np.nan
    ax1.plot(t, e_q, color='red', linewidth=2, label='QRS')
    e_t = ecg.copy(); e_t[~t_mask] = np.nan
    ax1.plot(t, e_t, color='green', linewidth=2, label='T-Wave')
    ax1.set_title(f"{pid} | Radar-Reconstructed ECG Analysis")
    ax1.legend(loc='upper right')
    
    ax2.plot(t, flow, 'k-', linewidth=1.5, label='Flow Velocity')
    for r in r_peaks:
        t_s, t_e = r+int(0.05*fs), r+int(0.35*fs)
        if t_e < len(t): ax2.axvspan(t[t_s], t[t_e], color='purple', alpha=0.1)
    ax2.set_ylabel("dICG/dt"); ax2.legend(); ax2.set_xlabel("Time (s)")
    
    plt.tight_layout()
    save_path = f"{save_dir}/{pid}_clinical_report.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved Plot: {save_path}")
    plt.close()

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True, help="Patient ID")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--maneuver", type=str, default="1_Resting")
    args = parser.parse_args()

    device = torch.device(cfg.device)
    save_dir = "clinical_reports"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading Model from {args.ckpt}...")
    cfg.in_channels = 4
    model = SimplifiedCASTECG(cfg).to(device)
    try:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        check_model_health(model) # <--- HEALTH CHECK
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
        
    model.eval()

    mat_path = Path(cfg.data_root) / args.id / f"{args.id}_{args.maneuver}.mat"
    print(f"Processing {args.id}...")
    radar_np = process_patient_data_on_the_fly(mat_path)

    print("Running Inference...")
    W = cfg.window_size_samples
    S = cfg.stride_samples
    L = radar_np.shape[1]
    
    recon_ecg = np.zeros(L)
    recon_flow = np.zeros(L)
    counts = np.zeros(L)
    hann = scipy.signal.windows.hann(W)

    with torch.no_grad():
        for i in range(0, L-W, S):
            x = torch.from_numpy(radar_np[:, i:i+W]).unsqueeze(0).float().to(device)
            # Sanity check input to model
            if torch.isnan(x).any():
                print(f"NaNs detected in input window {i}, skipping...")
                continue
                
            out = model(x)
            
            ecg_win = out['ecg'][0, 1].cpu().numpy()
            flow_win = out['dicg'][0, 0].cpu().numpy()
            
            recon_ecg[i:i+W] += ecg_win * hann
            recon_flow[i:i+W] += flow_win * hann
            counts[i:i+W] += hann

    mask = counts > 0
    recon_ecg[mask] /= counts[mask]
    recon_flow[mask] /= counts[mask]
    
    # Handle NaNs in output
    recon_ecg = np.nan_to_num(recon_ecg)
    recon_flow = np.nan_to_num(recon_flow)
    
    final_ecg = recon_ecg[slice(W//2, -W//2)]
    final_flow = recon_flow[slice(W//2, -W//2)]

    print(f"DEBUG: Output Range: {final_ecg.min():.4f} to {final_ecg.max():.4f}")

    print("Generating Report...")
    r_peaks = detect_peaks_robust(final_ecg, cfg.fs)
    
    if len(r_peaks) > 1:
        rr = np.diff(r_peaks) / cfg.fs
        hr = 60.0 / np.mean(rr)
        hrv = np.sqrt(np.mean(np.diff(rr*1000)**2))
    else:
        hr, hrv = 0.0, 0.0

    hemo = calculate_hemodynamics(final_flow, r_peaks, cfg.fs)
    
    report = f"""
    CLINICAL REPORT: {args.id}
    ==========================
    Heart Rate:    {hr:.1f} bpm (Peaks: {len(r_peaks)})
    HRV (RMSSD):   {hrv:.1f} ms
    Contractility: {hemo['Contractility']:.2f}
    """
    print(report)
    
    plot_colored_ecg(args.id, final_ecg[:1280], final_flow[:1280], r_peaks[r_peaks < 1280], cfg.fs, save_dir)

if __name__ == "__main__":
    main()