"""
Preprocess: 4-Channel Radar Input + 7 Tasks
Input Channels:
0: Phase (Heart Band 0.8-8Hz)
1: Mag (Heart Band 0.8-8Hz)
2: Phase (Resp Band 0.05-0.8Hz)
3: Mag (Resp Band 0.05-0.8Hz)
"""
import os
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly, butter, filtfilt, iirnotch
from pathlib import Path
import h5py
from tqdm import tqdm
import sys

try:
    from configs.config import cfg 
except ImportError:
    sys.exit("Error: Could not import 'configs.config'.")

# --- CONSTANTS ---
TARGET_FS = cfg.fs
NOTCH_FREQ = 50.0       
BASELINE_POLY_ORDER = 5 

# --- CRITICAL: SEPARATE BANDS ---
HEART_BAND = [0.8, 8.0]  # Clean for ECG
RESP_BAND_RADAR = [0.05, 0.8] # Clean for Breathing

# Target Bands
ECG_BAND = [0.5, 40.0]
ICG_BAND = [0.5, 20.0]
BP_BAND  = [0.05, 10.0]
RESP_CUTOFF = 0.5 

ALL_PATIENT_IDS = [f"GDN{str(i).zfill(4)}" for i in range(1, 31)]
TRAIN_IDS, VAL_IDS, TEST_IDS = ALL_PATIENT_IDS[:26], ALL_PATIENT_IDS[26:27], ALL_PATIENT_IDS[27:]

def apply_processing(signal, fs, band=None, notch=False, baseline=True):
    if len(signal) == 0: return signal
    if notch:
        b, a = iirnotch(NOTCH_FREQ, 30.0, fs)
        signal = filtfilt(b, a, signal)
    if band:
        b, a = butter(4, [band[0]/(0.5*fs), band[1]/(0.5*fs)], btype='band')
        signal = filtfilt(b, a, signal)
    if baseline:
        x = np.arange(len(signal))
        p = np.polyfit(x, signal, BASELINE_POLY_ORDER)
        signal = signal - np.polyval(p, x)
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def process_respiration(signal, fs_orig, target_fs):
    """Low-Pass for Ground Truth Resp"""
    if len(signal) == 0: return signal
    sig_res = resample_poly(signal, int(target_fs), int(fs_orig))
    nyq = 0.5 * target_fs
    b, a = butter(4, RESP_CUTOFF / nyq, btype='low')
    sig_filt = filtfilt(b, a, sig_res)
    sig_centered = sig_filt - np.mean(sig_filt)
    return sig_centered / (np.std(sig_centered) + 1e-8)

def derive_leads(I, II):
    III = II - I
    aVR = -0.5 * (I + II)
    aVL = I - 0.5 * II
    aVF = II - 0.5 * I
    return III, aVR, aVL, aVF

def process_patient_data(mat_path):
    d = sio.loadmat(mat_path)
    has_bp = 'tfm_bp' in d
    has_strain = 'tfm_intervention' in d
    
    # 1. Radar (Create 4 Channels)
    fs_radar = d.get('fs_radar', [[2000]])[0, 0]
    up, down = int(TARGET_FS), int(fs_radar)
    i_r = resample_poly(d['radar_i'].flatten(), up, down)
    q_r = resample_poly(d['radar_q'].flatten(), up, down)
    
    # Basic Phase/Mag calculation
    raw_phase = np.unwrap(np.arctan2(q_r, i_r))
    raw_mag = np.sqrt(i_r**2 + q_r**2)
    
    # Set A: Heart Band (0.8 - 8.0 Hz) -> For ECG, ICG, BP
    p_heart = apply_processing(raw_phase, TARGET_FS, HEART_BAND, notch=True, baseline=True)
    m_heart = apply_processing(raw_mag, TARGET_FS, HEART_BAND, notch=True, baseline=True)
    
    # Set B: Resp Band (0.05 - 0.8 Hz) -> For Respiration, Strain
    # Note: No Polynomial Baseline here! We want the slow wave.
    p_resp = apply_processing(raw_phase, TARGET_FS, RESP_BAND_RADAR, notch=False, baseline=False)
    m_resp = apply_processing(raw_mag, TARGET_FS, RESP_BAND_RADAR, notch=False, baseline=False)
    
    # Stack -> [4, L]
    radar = np.stack([p_heart, m_heart, p_resp, m_resp])

    # 2. ECG
    e1 = apply_processing(resample_poly(d['tfm_ecg1'].flatten(), up, down), TARGET_FS, ECG_BAND)
    e2 = apply_processing(resample_poly(d['tfm_ecg2'].flatten(), up, down), TARGET_FS, ECG_BAND)
    e3, avr, avl, avf = derive_leads(e1, e2)
    ecg_stacked = np.stack([e1, e2, e3, avr, avl, avf]) 

    # 3. ICG
    fs_icg = d.get('fs_icg', [[1000]])[0, 0]
    icg_raw = resample_poly(d['tfm_icg'].flatten(), int(TARGET_FS), int(fs_icg))
    icg = apply_processing(icg_raw, TARGET_FS, ICG_BAND)
    dicg = np.gradient(icg)
    dicg = (dicg - np.mean(dicg)) / (np.std(dicg) + 1e-8)

    # 4. Resp
    fs_z0 = d.get('fs_z0', [[100]])[0, 0]
    if 'tfm_z0' in d:
        z0_raw = d['tfm_z0'].flatten()
        if np.isnan(z0_raw).any() or np.std(z0_raw) < 1e-6:
            resp = np.zeros_like(e1)
        else:
            resp = process_respiration(z0_raw, fs_z0, TARGET_FS)
            if len(resp) != len(e1): 
                resp = np.interp(np.linspace(0,1,len(e1)), np.linspace(0,1,len(resp)), resp)
    else:
        resp = np.zeros_like(e1)

    # 5. BP & Strain
    if has_bp:
        fs_bp = d.get('fs_bp', [[200]])[0, 0]
        bp = apply_processing(resample_poly(d['tfm_bp'].flatten(), int(TARGET_FS), int(fs_bp)), TARGET_FS, BP_BAND)
    else:
        bp = np.zeros_like(e1)
        
    if has_strain:
        fs_int = d.get('fs_intervention', [[2000]])[0, 0]
        strain = apply_processing(resample_poly(d['tfm_intervention'].flatten(), int(TARGET_FS), int(fs_int)), TARGET_FS)
    else:
        strain = np.zeros_like(e1)

    L = min(radar.shape[1], ecg_stacked.shape[1], len(icg), len(resp), len(bp), len(strain))
    
    # Flags
    has_resp_signal = 1.0 if np.std(resp) > 0.1 else 0.0
    flags = np.array([1.0 if has_bp else 0.0, 1.0 if has_strain else 0.0, has_resp_signal])

    return (
        radar[:, :L].astype(np.float32), 
        ecg_stacked[:, :L].astype(np.float32), 
        icg[:L].astype(np.float32),
        dicg[:L].astype(np.float32),
        bp[:L].astype(np.float32),
        strain[:L].astype(np.float32),
        resp[:L].astype(np.float32),
        flags.astype(np.float32)
    )

def main():
    maneuver_info = cfg.maneuvers_to_load[0]
    m_code, m_name = maneuver_info
    output_file = cfg.h5_file_pattern.format(m=m_name)
    print(f"Generating 4-Channel Input Dataset: {output_file}")
    
    splits = {'train': TRAIN_IDS, 'val': VAL_IDS, 'test': TEST_IDS}
    
    # Write Split Info
    with open("dataset_split_info.txt", "w") as txt:
        txt.write("=== DATASET SPLIT INFO ===\n")
        txt.write(f"TRAIN IDs ({len(TRAIN_IDS)}): {', '.join(TRAIN_IDS)}\n")
        txt.write(f"VAL IDs   ({len(VAL_IDS)}): {', '.join(VAL_IDS)}\n")
        txt.write(f"TEST IDs  ({len(TEST_IDS)}): {', '.join(TEST_IDS)}\n")

    with h5py.File(output_file, 'w') as f:
        f.attrs['fs'] = TARGET_FS
        for split_name, patient_list in splits.items():
            print(f"\nProcessing {split_name} split...")
            data = {k: [] for k in ['r', 'e', 'i', 'di', 'b', 's', 'z', 'f']}
            for p_id in tqdm(patient_list):
                path = Path(cfg.data_root) / p_id / f"{p_id}_{m_code}_{m_name}.mat"
                if not path.exists(): continue
                try:
                    # r will now be [4, L]
                    r, e, i, di, b, s, z, fl = process_patient_data(path)
                    for k in range(0, r.shape[1] - cfg.window_size_samples, cfg.stride_samples):
                        data['r'].append(r[:, k:k+cfg.window_size_samples])
                        data['e'].append(e[:, k:k+cfg.window_size_samples])
                        data['i'].append(i[k:k+cfg.window_size_samples])
                        data['di'].append(di[k:k+cfg.window_size_samples])
                        data['b'].append(b[k:k+cfg.window_size_samples])
                        data['s'].append(s[k:k+cfg.window_size_samples])
                        data['z'].append(z[k:k+cfg.window_size_samples])
                        data['f'].append(fl)
                except Exception as ex: print(f"Err {p_id}: {ex}")
            
            if data['r']:
                f.create_dataset(f'{split_name}_radar', data=np.array(data['r']))
                f.create_dataset(f'{split_name}_ecg', data=np.array(data['e']))
                f.create_dataset(f'{split_name}_icg', data=np.array(data['i'])[:, np.newaxis, :])
                f.create_dataset(f'{split_name}_dicg', data=np.array(data['di'])[:, np.newaxis, :])
                f.create_dataset(f'{split_name}_bp', data=np.array(data['b'])[:, np.newaxis, :])
                f.create_dataset(f'{split_name}_strain', data=np.array(data['s'])[:, np.newaxis, :])
                f.create_dataset(f'{split_name}_resp', data=np.array(data['z'])[:, np.newaxis, :])
                f.create_dataset(f'{split_name}_flags', data=np.array(data['f']))
                f.create_dataset(f'{split_name}_mask', data=np.ones((len(data['e']), cfg.window_size_samples)))
                print(f"  -> Saved {len(data['r'])} windows.")

if __name__ == "__main__":
    main()