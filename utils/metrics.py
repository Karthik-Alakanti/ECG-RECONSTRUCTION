import numpy as np
from scipy.stats import pearsonr

def get_pcc(pred, true):
    """Pearson Correlation Coefficient"""
    if np.std(pred) < 1e-6 or np.std(true) < 1e-6:
        return 0.0
    return pearsonr(pred.flatten(), true.flatten())[0]

def get_temporal_correlation(pred, true):
    """Temporal correlation: normalized PCC of time-aligned signals"""
    if pred.ndim == 2:
        # Multi-channel: average across channels
        correlations = []
        for ch in range(pred.shape[0]):
            if np.std(pred[ch]) < 1e-6 or np.std(true[ch]) < 1e-6:
                correlations.append(0.0)
            else:
                correlations.append(pearsonr(pred[ch].flatten(), true[ch].flatten())[0])
        return np.mean(correlations)
    else:
        # Single channel
        if np.std(pred) < 1e-6 or np.std(true) < 1e-6:
            return 0.0
        return pearsonr(pred.flatten(), true.flatten())[0]

def get_spectral_correlation(pred, true, fs=128.0):
    """Spectral correlation using FFT-based normalized magnitude vectors"""
    if pred.ndim == 2:
        # Multi-channel: average across channels
        correlations = []
        for ch in range(pred.shape[0]):
            pred_fft = np.abs(np.fft.rfft(pred[ch]))
            true_fft = np.abs(np.fft.rfft(true[ch]))
            
            # L2 normalize
            pred_norm = pred_fft / (np.linalg.norm(pred_fft) + 1e-8)
            true_norm = true_fft / (np.linalg.norm(true_fft) + 1e-8)
            
            # Cosine similarity
            corr = np.dot(pred_norm, true_norm)
            correlations.append(corr)
        return np.mean(correlations)
    else:
        # Single channel
        pred_fft = np.abs(np.fft.rfft(pred))
        true_fft = np.abs(np.fft.rfft(true))
        
        pred_norm = pred_fft / (np.linalg.norm(pred_fft) + 1e-8)
        true_norm = true_fft / (np.linalg.norm(true_fft) + 1e-8)
        
        return np.dot(pred_norm, true_norm)

def get_mae(pred, true):
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - true))

def get_rmse(pred, true):
    """Root Mean Square Error"""
    return np.sqrt(np.mean((pred - true)**2))

def get_bp_metrics(pred_bp, true_bp):
    """
    Extracts Systolic (Max) and Diastolic (Min) from BP Waveforms
    Returns: MAE_SBP, MAE_DBP
    """
    pred_sbp = np.max(pred_bp)
    pred_dbp = np.min(pred_bp)
    
    true_sbp = np.max(true_bp)
    true_dbp = np.min(true_bp)
    
    sbp_error = np.abs(pred_sbp - true_sbp)
    dbp_error = np.abs(pred_dbp - true_dbp)
    
    return sbp_error, dbp_error

def get_resp_rate(signal, fs):
    """Calculates Respiration Rate (RPM) using FFT"""
    if np.std(signal) < 1e-6: return 0.0
    
    n = len(signal)
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    
    # Resp band 0.1 - 0.8 Hz
    mask = (freqs >= 0.1) & (freqs <= 0.8)
    if not mask.any(): return 0.0
    
    peak_idx = np.argmax(np.abs(fft) * mask)
    peak_freq = freqs[peak_idx]
    
    return peak_freq * 60.0 # Convert to RPM

def calculate_all_metrics(preds, targets, fs):
    """
    Master function to compute metrics for one batch/window.
    Uses temporal and spectral correlations (aligned with loss functions) instead of raw PCC.
    """
    m = {}
    
    # 1. ECG (Temporal + Spectral Correlation)
    p_ecg = preds['ecg_6lead'] if 'ecg_6lead' in preds else preds['ecg']
    t_ecg = targets['ecg_6lead'] if 'ecg_6lead' in targets else targets['ecg']
    
    m['ECG_Temporal_Corr'] = get_temporal_correlation(p_ecg, t_ecg)
    m['ECG_Spectral_Corr'] = get_spectral_correlation(p_ecg, t_ecg, fs)
    m['ECG_PCC'] = np.mean([get_pcc(p_ecg[i], t_ecg[i]) for i in range(len(p_ecg))])
    m['ECG_RMSE'] = get_rmse(p_ecg, t_ecg)
    
    # 2. BP (Temporal + Spectral Correlation + Systolic/Diastolic Error)
    sbp_err, dbp_err = get_bp_metrics(preds['bp'], targets['bp'])
    m['BP_SBP_MAE'] = sbp_err
    m['BP_DBP_MAE'] = dbp_err
    m['BP_Temporal_Corr'] = get_temporal_correlation(preds['bp'], targets['bp'])
    m['BP_Spectral_Corr'] = get_spectral_correlation(preds['bp'], targets['bp'], fs)
    m['BP_PCC'] = get_pcc(preds['bp'], targets['bp'])
    
    # 3. ICG & Flow (Temporal + Spectral Correlation)
    m['ICG_Temporal_Corr'] = get_temporal_correlation(preds['icg'], targets['icg'])
    m['ICG_Spectral_Corr'] = get_spectral_correlation(preds['icg'], targets['icg'], fs)
    m['ICG_PCC'] = get_pcc(preds['icg'], targets['icg'])
    
    m['Flow_Temporal_Corr'] = get_temporal_correlation(preds['dicg'], targets['dicg'])
    m['Flow_Spectral_Corr'] = get_spectral_correlation(preds['dicg'], targets['dicg'], fs)
    m['Flow_PCC'] = get_pcc(preds['dicg'], targets['dicg'])
    
    
    # 4. Strain (MAE)
    m['Strain_MAE'] = get_mae(preds['strain'], targets['strain'])
    
    # 5. Resp (Rate Error)
    pred_rpm = get_resp_rate(preds['resp'], fs)
    true_rpm = get_resp_rate(targets['resp'], fs)
    m['Resp_RPM_Error'] = np.abs(pred_rpm - true_rpm) # <--- Matches test.py
    m['Resp_PCC'] = get_pcc(preds['resp'], targets['resp'])
    
    return m